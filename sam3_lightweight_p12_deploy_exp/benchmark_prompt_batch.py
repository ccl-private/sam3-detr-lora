"""比较7个固定提示串行与单次批量DETR的速度、显存和十图IoU。"""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageDraw
from deploy import HERE, EXP, PROMPTS, load_package
from runtime_optimizations import CachedEmptyGeometryProcessor
from sam3.model import box_ops
from sam3.model.data_misc import FindStage, interpolate


@torch.inference_mode()
def batch_prompts(proc, image, prompts):
    state=proc.set_image(image)
    n=len(prompts); device=proc.device
    state['backbone_out'].update(proc.model.backbone.forward_text(prompts,device=device))
    find=FindStage(img_ids=torch.zeros(n,dtype=torch.long,device=device),
        text_ids=torch.arange(n,device=device),input_boxes=None,input_boxes_mask=None,
        input_boxes_label=None,input_points=None,input_points_mask=None)
    raw=proc.model.forward_grounding(state['backbone_out'],find,None,
                                      proc.model._get_dummy_prompt(n))
    probs=raw['pred_logits'].sigmoid().squeeze(-1)*raw['presence_logit_dec'].sigmoid()
    results=[]
    for i in range(n):
        keep=probs[i]>proc.confidence_threshold
        masks=interpolate(raw['pred_masks'][i,keep].unsqueeze(1),
            (state['original_height'],state['original_width']),mode='bilinear',
            align_corners=False).sigmoid()
        boxes=box_ops.box_cxcywh_to_xyxy(raw['pred_boxes'][i,keep])
        scale=torch.tensor([state['original_width'],state['original_height']]*2,device=device)
        results.append({'masks':masks>.5,'boxes':boxes*scale,'scores':probs[i,keep]})
    return results


@torch.inference_mode()
def serial_prompts(proc,image,prompts):
    state=proc.set_image(image); results=[]
    for prompt in prompts:
        out=proc.set_text_prompt(prompt,state)
        results.append({k:out[k].clone() for k in ('masks','boxes','scores')})
    return results


def timed(fn,warmup,repeats):
    for _ in range(warmup): fn()
    values=[]
    for _ in range(repeats):
        torch.cuda.synchronize(); start=time.perf_counter(); fn(); torch.cuda.synchronize()
        values.append((time.perf_counter()-start)*1000)
    return {'mean_ms':float(np.mean(values)),'p50_ms':float(np.percentile(values,50)),
            'p95_ms':float(np.percentile(values,95))}


def truth_masks(path,image):
    masks=[Image.new('1',image.size) for _ in PROMPTS]
    for line in path.with_suffix('.txt').read_text().splitlines():
        fields=line.split()
        if len(fields)<7: continue
        cid=int(fields[0]); xy=list(map(float,fields[1:]))
        if 0<=cid<len(masks):
            ImageDraw.Draw(masks[cid]).polygon([(xy[j]*image.width,xy[j+1]*image.height)
                for j in range(0,len(xy),2)],fill=1)
    return [np.asarray(x,dtype=bool) for x in masks]


@torch.inference_mode()
def main(a):
    model=load_package(HERE/'weights/p12_fixed_vocab_fp16.pt').float().cuda().eval()
    proc=CachedEmptyGeometryProcessor(model,device='cuda:0',confidence_threshold=.5)
    paths=[Path(x) for x in json.loads((EXP/'tests/output/p12_query_set_best_first10_threshold_05/summary.json').read_text())['images'][:a.limit]]
    totals={'serial':np.zeros((7,3),dtype=np.int64),'batch':np.zeros((7,3),dtype=np.int64)}
    disagreement=np.zeros(7,dtype=np.int64); union_pixels=np.zeros(7,dtype=np.int64)
    count_changes=np.zeros(7,dtype=np.int64)
    with torch.autocast('cuda',dtype=torch.float16):
        for path in paths:
            image=Image.open(path).convert('RGB'); truth=truth_masks(path,image)
            outputs={'serial':serial_prompts(proc,image,PROMPTS),
                     'batch':batch_prompts(proc,image,PROMPTS)}
            for cid in range(7):
                predictions={}
                for mode in outputs:
                    masks=outputs[mode][cid]['masks']
                    pred=masks.reshape(-1,image.height,image.width).any(0).cpu().numpy() if len(masks) else np.zeros((image.height,image.width),bool)
                    predictions[mode]=pred; gt=truth[cid]
                    totals[mode][cid]+=np.array([(pred & gt).sum(),pred.sum(),gt.sum()])
                disagreement[cid]+=np.logical_xor(predictions['serial'],predictions['batch']).sum()
                union_pixels[cid]+=np.logical_or(predictions['serial'],predictions['batch']).sum()
                count_changes[cid]+=len(outputs['serial'][cid]['scores'])!=len(outputs['batch'][cid]['scores'])
            print(path.name,flush=True)
        image=Image.open(paths[0]).convert('RGB')
        torch.cuda.reset_peak_memory_stats(); timing_serial=timed(lambda:serial_prompts(proc,image,PROMPTS),a.warmup,a.repeats)
        serial_peak=torch.cuda.max_memory_allocated()/2**20
        torch.cuda.reset_peak_memory_stats(); timing_batch=timed(lambda:batch_prompts(proc,image,PROMPTS),a.warmup,a.repeats)
        batch_peak=torch.cuda.max_memory_allocated()/2**20
    def metrics(mode):
        return {p:{'iou':float(x[0]/max(x[1]+x[2]-x[0],1)),
                   'recall':float(x[0]/max(x[2],1)),'precision':float(x[0]/max(x[1],1))}
                for p,x in zip(PROMPTS,totals[mode])}
    report={'gpu':torch.cuda.get_device_name(),'precision':'FP16 autocast, FP32 parameters',
        'images':len(paths),'serial':{'metrics':metrics('serial'),'timing':timing_serial,'peak_mib':serial_peak},
        'batch':{'metrics':metrics('batch'),'timing':timing_batch,'peak_mib':batch_peak},
        'foreground_disagreement':{p:float(disagreement[i]/max(union_pixels[i],1)) for i,p in enumerate(PROMPTS)},
        'images_with_detection_count_change':{p:int(count_changes[i]) for i,p in enumerate(PROMPTS)},
        'warmup':a.warmup,'repeats':a.repeats}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--limit',type=int,default=10); p.add_argument('--warmup',type=int,default=10)
    p.add_argument('--repeats',type=int,default=50)
    p.add_argument('--output',type=Path,default=HERE/'tests/output/prompt_batch/report.json')
    a=p.parse_args(); torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    main(a)
