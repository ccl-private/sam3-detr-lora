"""将单提示延时严格拆成预处理、神经网络前向和后处理。"""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import v2
from deploy import HERE, EXP, PROMPTS, load_package
from runtime_optimizations import CachedEmptyGeometryProcessor
from sam3.model import box_ops
from sam3.model.data_misc import interpolate


def timed(fn,warmup,repeats):
    for _ in range(warmup): fn()
    values=[]
    for _ in range(repeats):
        torch.cuda.synchronize(); start=time.perf_counter(); fn(); torch.cuda.synchronize()
        values.append((time.perf_counter()-start)*1000)
    return {'mean_ms':float(np.mean(values)),'p50_ms':float(np.percentile(values,50)),
            'p95_ms':float(np.percentile(values,95))}


@torch.inference_mode()
def main(a):
    model=load_package(HERE/'weights/p12_fixed_vocab_fp16.pt').float().cuda().eval()
    proc=CachedEmptyGeometryProcessor(model,device='cuda:0',confidence_threshold=.5)
    # 单提示每帧只调用一次几何编码，不使用跨文本缓存。
    proc.cache_enabled=False
    path=Path(json.loads((EXP/'tests/output/p12_query_set_best_first10_threshold_05/summary.json').read_text())['images'][0])
    # 文件读取、图片解码和RGB转换全部在计时区间外。
    image=Image.open(path).convert('RGB')
    cpu_tensor=v2.functional.to_image(image)
    gpu_tensor=cpu_tensor.cuda()
    ready=proc.transform(gpu_tensor).unsqueeze(0)
    text=model.backbone.forward_text([PROMPTS[0]],device='cuda:0')

    def preprocess_pil():
        return proc.transform(v2.functional.to_image(image).to('cuda:0')).unsqueeze(0)
    def preprocess_gpu():
        return proc.transform(gpu_tensor).unsqueeze(0)
    def image_forward():
        return model.backbone.forward_image(ready)
    backbone=image_forward(); backbone.update(text)
    dummy=model._get_dummy_prompt()
    def text_lookup():
        return model.backbone.forward_text([PROMPTS[0]],device='cuda:0')
    def grounding():
        return model.forward_grounding(backbone,proc.find_stage,None,dummy)
    raw=grounding()
    def postprocess():
        probs=(raw['pred_logits'].sigmoid()*raw['presence_logit_dec'].sigmoid().unsqueeze(1)).squeeze(-1)
        keep=probs>proc.confidence_threshold
        boxes=box_ops.box_cxcywh_to_xyxy(raw['pred_boxes'][keep])
        scale=torch.tensor([image.width,image.height,image.width,image.height],device='cuda:0')
        boxes=boxes*scale[None]
        masks=interpolate(raw['pred_masks'][keep].unsqueeze(1),(image.height,image.width),
                          mode='bilinear',align_corners=False).sigmoid()>.5
        return masks,boxes,probs[keep]
    def neural_forward():
        bo=model.backbone.forward_image(ready); bo.update(text_lookup())
        return model.forward_grounding(bo,proc.find_stage,None,dummy)
    def tensor_pipeline():
        network_input=preprocess_gpu()
        bo=model.backbone.forward_image(network_input); bo.update(text_lookup())
        out=model.forward_grounding(bo,proc.find_stage,None,dummy)
        probs=(out['pred_logits'].sigmoid()*out['presence_logit_dec'].sigmoid().unsqueeze(1)).squeeze(-1)
        keep=probs>proc.confidence_threshold
        return interpolate(out['pred_masks'][keep].unsqueeze(1),(image.height,image.width),
                           mode='bilinear',align_corners=False).sigmoid()>.5
    def pil_pipeline():
        network_input=preprocess_pil()
        bo=model.backbone.forward_image(network_input); bo.update(text_lookup())
        out=model.forward_grounding(bo,proc.find_stage,None,dummy)
        probs=(out['pred_logits'].sigmoid()*out['presence_logit_dec'].sigmoid().unsqueeze(1)).squeeze(-1)
        keep=probs>proc.confidence_threshold
        return interpolate(out['pred_masks'][keep].unsqueeze(1),(image.height,image.width),
                           mode='bilinear',align_corners=False).sigmoid()>.5
    functions={'preprocess_from_pil_object':preprocess_pil,
        'preprocess_from_gpu_uint8_tensor':preprocess_gpu,
        'image_network':image_forward,'fixed_text_lookup':text_lookup,
        'grounding_network':grounding,'postprocess_to_original_masks':postprocess,
        'neural_network_total':neural_forward,
        'tensor_pipeline_excluding_load_decode':tensor_pipeline,
        'pil_pipeline_excluding_load_decode':pil_pipeline}
    with torch.autocast('cuda',dtype=torch.float16):
        report={name:timed(fn,a.warmup,a.repeats) for name,fn in functions.items()}
    result={'gpu':torch.cuda.get_device_name(),'precision':'FP16 autocast, FP32 parameters',
        'image':str(path),'input_size':list(image.size),'prompt':PROMPTS[0],
        'excluded':['模型构建','权重读取','权重传GPU','磁盘读图','图片解码','首次编译'],
        'warmup':a.warmup,'repeats':a.repeats,'timing':report,
        'note':'各阶段独立同步计时，分项之和不应替代组合路径实测。'}
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--warmup',type=int,default=20)
    p.add_argument('--repeats',type=int,default=100)
    p.add_argument('--output',type=Path,default=HERE/'tests/output/latency_stages/report.json')
    a=p.parse_args(); torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    main(a)
