"""固定白实线单图TensorRT推理，保存预测覆盖图与实例mask/框/分数。"""
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import v2

from tensorrt_runtime import EngineRunner


@torch.inference_mode()
def main(args):
    torch.set_num_threads(4)
    runner = EngineRunner(args.engine)
    with Image.open(args.image) as source:
        image = source.convert('RGB')
    transform = v2.Compose([v2.ToDtype(torch.uint8, scale=True),
        v2.Resize((1008, 1008)), v2.ToDtype(torch.float32, scale=True),
        v2.Normalize([0.5]*3, [0.5]*3)])
    tensor = transform(v2.functional.to_image(image).cuda()).unsqueeze(0)
    logits, presence, boxes, masks = runner(tensor)
    scores = (logits.sigmoid() * presence.sigmoid().unsqueeze(1)).squeeze(-1)
    keep = scores > args.threshold
    scores, boxes, masks = scores[keep], boxes[keep], masks[keep]
    boxes = torch.cat((boxes[:, :2]-boxes[:, 2:]/2,
                       boxes[:, :2]+boxes[:, 2:]/2), dim=-1)
    boxes *= boxes.new_tensor([image.width, image.height, image.width, image.height])
    result = [torch.nn.functional.interpolate(mask[None, None].float(),
        (image.height, image.width), mode='bilinear', align_corners=False)[0].gt(0).cpu().numpy()
        for mask in masks]
    masks_array = np.stack(result) if result else np.zeros((0, 1, image.height, image.width), dtype=bool)
    canvas = np.asarray(image).copy()
    union = masks_array.any(axis=(0, 1))
    canvas[union] = (canvas[union]*0.55+np.array([0, 220, 255])*0.45).astype(np.uint8)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas).save(args.output)
    np.savez_compressed(args.output.with_suffix('.npz'), masks=masks_array,
                        scores=scores.cpu().numpy(), boxes=boxes.cpu().numpy())
    print(f'固定白实线检测数={len(scores)}，预测图={args.output}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engine', type=Path, required=True)
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--threshold', type=float, default=0.5)
    main(parser.parse_args())
