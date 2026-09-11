"""验证空几何缓存的输出等价性，并拆分PIL预处理开销。"""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import v2
from deploy import HERE, EXP, PROMPTS, load_package, Sam3Processor, run
from runtime_optimizations import CachedEmptyGeometryProcessor


def timed(fn, warmup, repeats):
    for _ in range(warmup): fn()
    values=[]
    for _ in range(repeats):
        torch.cuda.synchronize(); start=time.perf_counter(); fn(); torch.cuda.synchronize()
        values.append((time.perf_counter()-start)*1000)
    return {'mean_ms':float(np.mean(values)), 'p50_ms':float(np.percentile(values,50)),
            'p95_ms':float(np.percentile(values,95))}


@torch.inference_mode()
def main(a):
    model=load_package(HERE/'weights/p12_fixed_vocab_fp16.pt').float().cuda().eval()
    baseline=Sam3Processor(model,device='cuda:0',confidence_threshold=.5)
    manifest=json.loads((EXP/'tests/output/p12_query_set_best_first10_threshold_05/summary.json').read_text())
    images=[Image.open(p).convert('RGB') for p in manifest['images'][:a.limit]]
    raw={}
    forward=model.forward_grounding
    def capture(*args,**kwargs):
        out=forward(*args,**kwargs)
        raw['value']={k:v.detach().clone() for k,v in out.items() if k in
                      ('pred_logits','presence_logit_dec','pred_boxes','pred_masks')}
        return out
    model.forward_grounding=capture
    optimized=CachedEmptyGeometryProcessor(model,device='cuda:0',confidence_threshold=.5)
    comparisons=[]
    with torch.autocast('cuda',dtype=torch.float16):
        for image in images:
            optimized.cache_enabled=False
            state=baseline.set_image(image)
            for prompt in PROMPTS:
                baseline.set_text_prompt(prompt,state)
                expected={k:v.clone() for k,v in raw['value'].items()}
                optimized._empty_geometry_cache=None
                optimized.cache_enabled=True
                # 复用完全相同的backbone_out，排除图像前向浮点差异。
                test_state={k:v for k,v in state.items() if k not in ('geometric_prompt','masks_logits','masks','boxes','scores')}
                optimized.set_text_prompt(prompt,test_state)
                actual=raw['value']
                comparisons.append({k:float((expected[k]-actual[k]).abs().max()) for k in expected})
                optimized.cache_enabled=False
        image=images[0]
        timing={}
        for n in (1,7):
            optimized.cache_enabled=False
            timing[f'baseline_{n}']=timed(lambda:run(baseline,image,PROMPTS[:n]),a.warmup,a.repeats)
            optimized.cache_enabled=True
            timing[f'cached_{n}']=timed(lambda:run(optimized,image,PROMPTS[:n]),a.warmup,a.repeats)
        # 当前set_image路径的输入准备拆分：PIL转CPU张量、传原始4K到GPU、GPU变换、图像网络。
        cpu_tensor=v2.functional.to_image(image)
        transform=baseline.transform
        def pil_to_cpu(): v2.functional.to_image(image)
        timing['pil_to_cpu_tensor']=timed(pil_to_cpu,a.warmup,a.repeats)
        def cpu_to_gpu(): cpu_tensor.to('cuda:0')
        timing['raw_cpu_to_gpu']=timed(cpu_to_gpu,a.warmup,a.repeats)
        gpu_tensor=cpu_tensor.to('cuda:0')
        timing['gpu_resize_normalize']=timed(lambda:transform(gpu_tensor),a.warmup,a.repeats)
        ready=transform(gpu_tensor).unsqueeze(0)
        timing['image_backbone_ready_tensor']=timed(lambda:model.backbone.forward_image(ready),a.warmup,a.repeats)
        optimized.cache_enabled=True
        timing['cpu_tensor_cached_1']=timed(lambda:run(optimized,cpu_tensor,PROMPTS[:1]),a.warmup,a.repeats)
        timing['gpu_tensor_cached_1']=timed(lambda:run(optimized,gpu_tensor,PROMPTS[:1]),a.warmup,a.repeats)
        timing['gpu_tensor_cached_7']=timed(lambda:run(optimized,gpu_tensor,PROMPTS),a.warmup,a.repeats)
    model.forward_grounding=forward
    max_abs={k:max(row[k] for row in comparisons) for k in comparisons[0]}
    report={'gpu':torch.cuda.get_device_name(),'precision':'FP16 autocast, FP32 parameters',
            'images':len(images),'prompts':len(comparisons),'raw_max_abs':max_abs,
            'exact':all(value==0 for value in max_abs.values()),'timing':timing,
            'warmup':a.warmup,'repeats':a.repeats}
    output=a.output
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--limit',type=int,default=10)
    p.add_argument('--warmup',type=int,default=20)
    p.add_argument('--repeats',type=int,default=100)
    p.add_argument('--output',type=Path,
                   default=HERE/'tests/output/runtime_optimizations/report.json')
    a=p.parse_args(); torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    main(a)
