"""将固定白实线TensorRT引擎的单提示延时拆成预处理、引擎前向和后处理。"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import v2

from deploy import EXP, PROMPTS, Sam3Processor
from sam3.model import box_ops
from sam3.model.data_misc import interpolate

from tensorrt_runtime import EngineRunner


def timed(fn, warmup, repeats):
    """与benchmark_latency_stages.py相同的独立同步计时，便于两条路径对照。"""
    for _ in range(warmup):
        fn()
    values = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        values.append((time.perf_counter() - start) * 1000)
    return {'mean_ms': float(np.mean(values)), 'p50_ms': float(np.percentile(values, 50)),
            'p95_ms': float(np.percentile(values, 95))}


@torch.inference_mode()
def main(a):
    runner = EngineRunner(a.engine)
    # 只复用Sam3Processor的图像变换和阈值，引擎路径不加载PyTorch模型权重。
    proc = Sam3Processor(None, device='cuda:0', confidence_threshold=0.5)
    path = Path(json.loads((EXP / 'tests/output/p12_query_set_best_first10_threshold_05/summary.json').read_text())['images'][0])
    # 文件读取、图片解码和RGB转换全部在计时区间外。
    image = Image.open(path).convert('RGB')
    gpu_tensor = v2.functional.to_image(image).cuda()
    ready = proc.transform(gpu_tensor).unsqueeze(0)

    def preprocess_gpu():
        return proc.transform(gpu_tensor).unsqueeze(0)

    def engine_forward():
        return runner(ready)

    outputs = runner(ready)
    scale = torch.tensor([image.width, image.height, image.width, image.height], device='cuda:0')

    def postprocess():
        logits, presence, raw_boxes, raw_masks = outputs
        probs = (logits.sigmoid() * presence.sigmoid().unsqueeze(1)).squeeze(-1)
        keep = probs > proc.confidence_threshold
        boxes = box_ops.box_cxcywh_to_xyxy(raw_boxes[keep]) * scale[None]
        masks = interpolate(raw_masks[keep].unsqueeze(1), (image.height, image.width),
                            mode='bilinear', align_corners=False).sigmoid() > .5
        return masks, boxes, probs[keep]

    def engine_pipeline():
        network_input = preprocess_gpu()
        logits, presence, raw_boxes, raw_masks = runner(network_input)
        probs = (logits.sigmoid() * presence.sigmoid().unsqueeze(1)).squeeze(-1)
        keep = probs > proc.confidence_threshold
        return interpolate(raw_masks[keep].unsqueeze(1), (image.height, image.width),
                           mode='bilinear', align_corners=False).sigmoid() > .5

    functions = {'preprocess_from_gpu_uint8_tensor': preprocess_gpu,
                 'engine_forward': engine_forward,
                 'postprocess_to_original_masks': postprocess,
                 'engine_pipeline_excluding_load_decode': engine_pipeline}
    report = {name: timed(fn, a.warmup, a.repeats) for name, fn in functions.items()}
    result = {'engine': str(a.engine), 'gpu': torch.cuda.get_device_name(),
              'image': str(path), 'input_size': list(image.size), 'prompt': PROMPTS[0],
              'excluded': ['引擎反序列化', '磁盘读图', '图片解码', '首次编译'],
              'warmup': a.warmup, 'repeats': a.repeats, 'timing': report,
              'note': '各阶段独立同步计时，分项之和不应替代组合路径实测；'
                      '预处理和后处理在FP32下运行，不含原PyTorch路径的FP16 autocast。'}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--engine', type=Path, required=True)
    p.add_argument('--warmup', type=int, default=20)
    p.add_argument('--repeats', type=int, default=100)
    p.add_argument('--output', type=Path, default=Path(__file__).resolve().parent / 'tests/output/latency_stages/engine_report.json')
    a = p.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    main(a)
