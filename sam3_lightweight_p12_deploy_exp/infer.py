"""加载P12固定词表单文件并保存单图预测。颜色仅表示预测覆盖。"""
import argparse
from contextlib import nullcontext
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from deploy import load_package, PACKAGE, PROMPTS, Sam3Processor
from runtime_optimizations import CachedEmptyGeometryProcessor

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--image',type=Path,required=True)
    p.add_argument('--package',type=Path,default=PACKAGE)
    p.add_argument('--prompt',choices=PROMPTS,default=PROMPTS[0])
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--threshold',type=float,default=0.5)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--precision',choices=['fp32','fp16','bf16'],default='fp32')
    a=p.parse_args()
    torch.set_num_threads(4)
    model=load_package(a.package).float().to(a.device).eval()
    # 单提示行为不变；同一图扩展为多个固定文本时自动复用空几何编码。
    proc=CachedEmptyGeometryProcessor(model,device=a.device,confidence_threshold=a.threshold)
    image=Image.open(a.image).convert('RGB')
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    amp=nullcontext() if a.precision=='fp32' else torch.autocast('cuda',dtype={'fp16':torch.float16,'bf16':torch.bfloat16}[a.precision])
    with torch.inference_mode(),amp:
        state=proc.set_image(image)
        out=proc.set_text_prompt(a.prompt,state)
    canvas=np.asarray(image).copy()
    masks=out['masks'].cpu().numpy()
    if len(masks):
        union=masks.reshape(len(masks),image.height,image.width).any(0)
        canvas[union]=(canvas[union]*0.55+np.array([0,220,255])*0.45).astype(np.uint8)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    Image.fromarray(canvas).save(a.output)
    np.savez_compressed(a.output.with_suffix('.npz'),masks=masks,
        boxes=out['boxes'].cpu().numpy(),scores=out['scores'].cpu().numpy())
    print(f'检测数={len(masks)}，预测图={a.output}')

if __name__=='__main__': main()
