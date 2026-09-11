"""分开验证FP16存储舍入和混合精度计算，并记录真实标签IoU。"""
import argparse
import gc
import json
import time
from contextlib import nullcontext
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageDraw
from deploy import PACKAGE, HERE, EXP, PROMPTS, load_package, reference, digest, Sam3Processor, run, sync

HALF = HERE / 'weights/p12_fixed_vocab_fp16.pt'

def convert():
    if HALF.exists():
        raise FileExistsError(HALF)
    p=torch.load(PACKAGE,map_location='cpu',weights_only=True)
    for group in ('state_dict','text'):
        p[group]={k:v.half() if v.is_floating_point() else v for k,v in p[group].items()}
    p['storage_dtype']='float16'
    p['fp32_source_sha256']=digest(PACKAGE)
    torch.save(p,HALF)
    restored=load_package(HALF).float().state_dict()
    assert all(torch.equal(v.float(),restored[k].float()) for k,v in p['state_dict'].items())
    print('FP16存储包MiB:', HALF.stat().st_size/2**20, flush=True)

def context(mode):
    dtype={'half_amp':torch.float16,'original_bf16':torch.bfloat16}.get(mode)
    return torch.autocast('cuda',dtype=dtype) if dtype else nullcontext()

@torch.inference_mode()
def evaluate(a):
    mode=a.mode
    model=reference()[0] if mode=='original_bf16' else load_package(PACKAGE if mode=='fixed_fp32' else HALF)
    # 半精度存储与实际计算分开：FP32参数加载，autocast选择安全的算子精度。
    model=model.float().to(a.device).eval()
    proc=Sam3Processor(model,device=a.device,confidence_threshold=0.5)
    images=[Path(p) for p in json.loads(a.manifest.read_text())['images']]
    totals=np.zeros((7,3),dtype=np.int64)
    report={'mode':mode,'gpu':torch.cuda.get_device_name(a.device),'images':[], 'timing':{},
        'warmup':a.warmup,'repeats':a.repeats,'storage_mib':None if mode=='original_bf16' else
        (PACKAGE if mode=='fixed_fp32' else HALF).stat().st_size/2**20}
    output=HERE/'tests/output/precision'/mode
    output.mkdir(parents=True,exist_ok=True)
    for idx,path in enumerate(images):
        image=Image.open(path).convert('RGB')
        label=path.with_suffix('.txt')
        if not label.exists(): raise FileNotFoundError(label)
        gt=[Image.new('1',image.size) for _ in PROMPTS]
        for line in label.read_text().splitlines():
            f=line.split()
            if len(f)<7: continue
            cid=int(f[0]); xy=list(map(float,f[1:]))
            if 0<=cid<7:
                ImageDraw.Draw(gt[cid]).polygon([(xy[i]*image.width,xy[i+1]*image.height)
                    for i in range(0,len(xy),2)],fill=1)
        masks=[]; row=[]
        with context(mode):
            state=proc.set_image(image)
            for cid,prompt in enumerate(PROMPTS):
                out=proc.set_text_prompt(prompt,state)
                pred=out['masks'].reshape(-1,image.height,image.width).any(0).cpu().numpy() if len(out['masks']) else np.zeros((image.height,image.width),bool)
                truth=np.asarray(gt[cid],dtype=bool)
                totals[cid]+=np.array([(pred&truth).sum(),pred.sum(),truth.sum()])
                masks.append(pred)
                row.append({'prompt':prompt,'detections':len(out['scores']), 'max_score':float(out['scores'].max()) if len(out['scores']) else 0})
        np.savez_compressed(output/f'{idx}.npz',masks=np.stack(masks))
        report['images'].append({'path':str(path),'prompts':row})
        print(mode,idx,path.name,flush=True)
    report['metrics']={p:dict(iou=float(t[0]/max(t[1]+t[2]-t[0],1)),recall=float(t[0]/max(t[2],1)),
        precision=float(t[0]/max(t[1],1))) for p,t in zip(PROMPTS,totals)}
    report['mean_iou_white']=float(np.mean([report['metrics'][PROMPTS[i]]['iou'] for i in (0,2)]))
    image=Image.open(images[0]).convert('RGB')
    del state,out
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    for n in (1,7):
        with context(mode):
            for _ in range(a.warmup): run(proc,image,PROMPTS[:n])
            times=[]
            for _ in range(a.repeats):
                sync(); start=time.perf_counter(); run(proc,image,PROMPTS[:n]); sync()
                times.append((time.perf_counter()-start)*1000)
        report['timing'][str(n)]={'mean_ms':float(np.mean(times)),'p50_ms':float(np.percentile(times,50)),
            'p95_ms':float(np.percentile(times,95))}
    report['peak_allocated_mib']=torch.cuda.max_memory_allocated()/2**20
    (output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(mode,report['mean_iou_white'],report['timing'],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['convert','fixed_fp32','half_storage','half_amp','original_bf16'])
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--manifest',type=Path,default=EXP/'tests/output/p12_query_set_best_first10_threshold_05/summary.json')
    p.add_argument('--warmup',type=int,default=30); p.add_argument('--repeats',type=int,default=200)
    a=p.parse_args(); torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    convert() if a.mode=='convert' else evaluate(a)
