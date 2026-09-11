"""从道路标线训练拆分确定性抽取图片，生成ModelOpt所需的单输入校准数组。"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main(args: argparse.Namespace) -> None:
    paths = sorted(
        path for path in args.source.iterdir()
        if path.suffix.lower() in IMAGE_SUFFIXES and path.exists()
    )
    if len(paths) < args.samples:
        raise RuntimeError(f"可用图片只有{len(paths)}张，少于要求的{args.samples}张")

    # 均匀覆盖按文件名排序后的训练拆分，避免只抽到同一视频的连续帧。
    indices = np.linspace(0, len(paths) - 1, args.samples, dtype=np.int64)
    selected = [paths[int(index)] for index in indices]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    data = np.lib.format.open_memmap(
        args.output, mode="w+", dtype=np.float32,
        shape=(args.samples, 3, args.resolution, args.resolution),
    )
    records = []
    for index, path in enumerate(selected):
        with Image.open(path) as image:
            image = image.convert("RGB").resize(
                (args.resolution, args.resolution), Image.Resampling.BILINEAR
            )
            array = np.asarray(image, dtype=np.float32)
        data[index] = np.transpose(array / 127.5 - 1.0, (2, 0, 1))
        records.append({"index": index, "path": str(path), "resolved": str(path.resolve())})
        print(f"[{index + 1}/{args.samples}] {path.name}", flush=True)
    data.flush()
    del data

    report = {
        "format": "p12_int8_calibration_v1",
        "source_split": str(args.source),
        "source_image_count": len(paths),
        "selection": "按文件名排序后等间距抽取",
        "samples": args.samples,
        "shape": [args.samples, 3, args.resolution, args.resolution],
        "dtype": "float32",
        "normalization": "RGB / 127.5 - 1.0",
        "data": str(args.output),
        "data_sha256": sha256(args.output),
        "images": records,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({key: report[key] for key in (
        "samples", "shape", "data", "data_sha256"
    )}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path,
        default=Path("/slow_disk/ccl/data/roadline20251023/video_disjoint/train"),
    )
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--resolution", type=int, default=1008)
    parser.add_argument(
        "--output", type=Path,
        default=Path("sam3_lightweight_p12_deploy_exp/tests/output/onnx_int8/calibration_train32.npy"),
    )
    parser.add_argument(
        "--manifest", type=Path,
        default=Path("sam3_lightweight_p12_deploy_exp/tests/output/onnx_int8/calibration_train32.json"),
    )
    main(parser.parse_args())

