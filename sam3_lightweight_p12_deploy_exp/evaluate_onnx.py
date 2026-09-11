"""固定白实线ONNX/引擎对照PyTorch，统计原始输出误差及验证图真实标签IoU、Recall。"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torchvision.transforms.v2 import functional as VF

from deploy import EXP, PACKAGE, PROMPTS, Sam3Processor, load_package
from export_onnx import FixedSinglePromptNetwork

from tensorrt_runtime import EngineRunner, NAMES


def union_mask(outputs, size):
    logits, presence, _, masks = outputs
    scores = (logits.sigmoid() * presence.sigmoid().unsqueeze(1)).squeeze(-1)
    masks = masks[scores > 0.5]
    union = torch.zeros(size, dtype=torch.bool, device=logits.device)
    for mask in masks:
        union |= torch.nn.functional.interpolate(mask[None, None].float(), size, mode='bilinear', align_corners=False)[0, 0] > 0
    return union.cpu().numpy(), len(masks)


@torch.inference_mode()
def main(args):
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = load_package(PACKAGE).float().cuda().eval()
    wrapper = FixedSinglePromptNetwork(model, PROMPTS[0]).eval()
    proc = Sam3Processor(model, device='cuda')
    if args.model.suffix == '.engine':
        candidate = EngineRunner(args.model)
    else:
        import onnxruntime as ort
        ort.preload_dlls()
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        session = ort.InferenceSession(str(args.model), sess_options=options,
            providers=[('CUDAExecutionProvider', {'device_id': 0, 'use_tf32': 0}), 'CPUExecutionProvider'])
        if 'CUDAExecutionProvider' not in session.get_providers():
            raise RuntimeError('CUDAExecutionProvider未启用')
        def candidate(image):
            values = session.run(NAMES, {'image': image.cpu().numpy()})
            return [torch.from_numpy(value).cuda() for value in values]
    paths = json.loads(args.manifest.read_text())['images'][:args.samples]
    report = {'model': str(args.model), 'prompt': PROMPTS[0], 'images': [],
              'onnxruntime_use_tf32': False,
              'scope': '白实线固定单提示；原始PyTorch抗锯齿版本作为基准；不含性能测量'}
    totals = np.zeros((2, 3), dtype=np.int64)
    for path in paths:
        path = Path(path)
        with Image.open(path) as source:
            image = source.convert('RGB')
        tensor = proc.transform(VF.to_image(image).cuda()).unsqueeze(0)
        baseline = wrapper(tensor)
        actual = candidate(tensor)
        errors = {}
        for name, ref, out in zip(NAMES, baseline, actual):
            if ref.shape != out.shape or not torch.isfinite(out).all():
                raise RuntimeError(f'{name}: shape mismatch or nonfinite output')
            delta = (ref.float() - out.float()).abs()
            errors[name] = {'max_abs': delta.max().item(), 'mean_abs': delta.mean().item()}
        truth_image = Image.new('1', image.size)
        for line in path.with_suffix('.txt').read_text().splitlines():
            fields = line.split()
            if len(fields) >= 7 and int(fields[0]) == 0:
                xy = list(map(float, fields[1:]))
                ImageDraw.Draw(truth_image).polygon([(xy[i]*image.width, xy[i+1]*image.height)
                    for i in range(0, len(xy), 2)], fill=1)
        truth = np.asarray(truth_image, dtype=bool)
        masks = []
        counts = []
        stats = []
        for index, outputs in enumerate((baseline, actual)):
            mask, count = union_mask(outputs, (image.height, image.width))
            current = np.array([(mask & truth).sum(), mask.sum(), truth.sum()], dtype=np.int64)
            totals[index] += current
            stats.append(current)
            masks.append(mask)
            counts.append(count)
        row = {'path': str(path), 'errors': errors, 'detections': counts,
               'mask_recall': [float(s[0]/max(s[2], 1)) for s in stats],
               'mask_disagreement_pixels': int((masks[0] != masks[1]).sum()),
               'mask_pair_iou': float((masks[0] & masks[1]).sum()/max((masks[0] | masks[1]).sum(), 1))}
        report['images'].append(row)
        print(json.dumps(row), flush=True)
    report['metrics'] = {name: {'iou': float(t[0]/max(t[1]+t[2]-t[0], 1)),
                                'recall': float(t[0]/max(t[2], 1))}
                         for name, t in zip(('pytorch_fp32', 'candidate'), totals)}
    if args.model.suffix == '.engine' and args.repeats > 0:
        for _ in range(10):
            candidate(tensor)
        torch.cuda.synchronize()
        times = []
        for _ in range(args.repeats):
            start = time.perf_counter()
            candidate(tensor)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000)
        report['timing'] = {'scope': 'GPU归一化输入到GPU原始输出，含输入复制与输出clone，不含预处理及mask后处理',
            'gpu': torch.cuda.get_device_name(), 'repeats': args.repeats,
            'mean_ms': float(np.mean(times)), 'p50_ms': float(np.percentile(times, 50)),
            'p95_ms': float(np.percentile(times, 95))}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(report['metrics'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--samples', type=int, default=10)
    parser.add_argument('--repeats', type=int, default=0, help='引擎独占GPU时设置以测量延迟')
    parser.add_argument('--manifest', type=Path, default=EXP/'tests/output/p12_query_set_best_first10_threshold_05/summary.json')
    main(parser.parse_args())
