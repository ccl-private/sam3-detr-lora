"""评测无CUDA Graph的torch.compile grounding单提示路径。"""
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
from benchmark_prompt_batch import serial_prompts, truth_masks, timed


@torch.inference_mode()
def evaluate(proc,paths):
    totals=np.zeros((7,3),dtype=np.int64); unions=[]; counts=[]
    for path in paths:
        image=Image.open(path).convert('RGB'); truth=truth_masks(path,image)
        out=serial_prompts(proc,image,PROMPTS); row=[]; nrow=[]
        for cid,item in enumerate(out):
            masks=item['masks']
            pred=masks.reshape(-1,image.height,image.width).any(0).cpu().numpy() if len(masks) else np.zeros((image.height,image.width),bool)
            gt=truth[cid]; totals[cid]+=np.array([(pred & gt).sum(),pred.sum(),gt.sum()])
            row.append(pred); nrow.append(len(item['scores']))
        unions.append(row); counts.append(nrow); print(path.name,flush=True)
    return totals,unions,np.asarray(counts)


def metrics(totals):
    return {p:{'iou':float(x[0]/max(x[1]+x[2]-x[0],1)),
               'recall':float(x[0]/max(x[2],1)),'precision':float(x[0]/max(x[1],1))}
            for p,x in zip(PROMPTS,totals)}


@torch.inference_mode()
def main(a):
    model=load_package(HERE/'weights/p12_fixed_vocab_fp16.pt').float().cuda().eval()
    proc=CachedEmptyGeometryProcessor(model,device='cuda:0',confidence_threshold=.5)
    paths=[Path(x) for x in json.loads((EXP/'tests/output/p12_query_set_best_first10_threshold_05/summary.json').read_text())['images'][:a.limit]]
    with torch.autocast('cuda',dtype=torch.float16):
        eager_tot,eager_masks,eager_counts=evaluate(proc,paths)
        original=model.forward_grounding
        model.forward_grounding=torch.compile(original,fullgraph=False,
            options={'triton.cudagraphs':False})
        # 首次调用触发编译，不计入稳态计时。
        serial_prompts(proc,Image.open(paths[0]).convert('RGB'),PROMPTS[:1])
        compiled_tot,compiled_masks,compiled_counts=evaluate(proc,paths)
        disagreement=np.zeros(7,dtype=np.int64); union=np.zeros(7,dtype=np.int64)
        for i in range(len(paths)):
            for cid in range(7):
                disagreement[cid]+=np.logical_xor(eager_masks[i][cid],compiled_masks[i][cid]).sum()
                union[cid]+=np.logical_or(eager_masks[i][cid],compiled_masks[i][cid]).sum()
        image=Image.open(paths[0]).convert('RGB'); tensor=v2.functional.to_image(image).cuda()
        def once(): serial_prompts(proc,tensor,PROMPTS[:1])
        compiled_time=timed(once,a.warmup,a.repeats)
        model.forward_grounding=original
        eager_time=timed(once,a.warmup,a.repeats)
    report={'gpu':torch.cuda.get_device_name(),'precision':'FP16 autocast, FP32 parameters',
        'images':len(paths),'eager':{'metrics':metrics(eager_tot),'timing':eager_time},
        'compiled':{'metrics':metrics(compiled_tot),'timing':compiled_time},
        'foreground_disagreement':{p:float(disagreement[i]/max(union[i],1)) for i,p in enumerate(PROMPTS)},
        'images_with_detection_count_change':{p:int((eager_counts[:,i]!=compiled_counts[:,i]).sum()) for i,p in enumerate(PROMPTS)},
        'compile_options':{'fullgraph':False,'triton.cudagraphs':False},
        'warmup':a.warmup,'repeats':a.repeats}
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--limit',type=int,default=10)
    p.add_argument('--warmup',type=int,default=10); p.add_argument('--repeats',type=int,default=50)
    p.add_argument('--output',type=Path,default=HERE/'tests/output/compile/report.json')
    a=p.parse_args(); torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    main(a)
