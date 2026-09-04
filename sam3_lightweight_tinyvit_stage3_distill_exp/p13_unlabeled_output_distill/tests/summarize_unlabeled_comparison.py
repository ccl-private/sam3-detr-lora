#!/usr/bin/env python3
"""汇总P12/P13-A无标签网图对比，并生成便于浏览的缩略图索引。"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sheet-size", type=int, default=40)
    args = parser.parse_args()
    rows = []
    for part in sorted((args.output / "parts").glob("part_*.jsonl")):
        rows.extend(json.loads(line) for line in part.read_text().splitlines() if line.strip())
    rows.sort(key=lambda row: (row["split"], row["relative_path"]))
    aggregate = defaultdict(lambda: defaultdict(float))
    for row in rows:
        for item in row["prompts"]:
            key = (row["split"], item["prompt"])
            a = aggregate[key]
            a["images"] += 1
            a["p12_detections"] += item["p12"]["detections"]
            a["p13a_detections"] += item["p13a"]["detections"]
            a["p12_pixels"] += item["p12"]["pred_pixels"]
            a["p13a_pixels"] += item["p13a"]["pred_pixels"]
            a["agreement_iou_sum"] += item["agreement_iou"]
            a["p12_only_pixels"] += item["p12_only_pixels"]
            a["p13a_only_pixels"] += item["p13a_only_pixels"]
    summary = {"num_images": len(rows), "color_meaning": {
        "P12/P13-A panels": "类别颜色只表示模型预测区域，不代表正确或错误",
        "DIFF blue": "仅P12预测", "DIFF green": "两模型预测重合", "DIFF orange": "仅P13-A预测",
    }, "splits": {}}
    for (split, prompt), values in aggregate.items():
        values = dict(values)
        values["mean_per_image_agreement_iou"] = values.pop("agreement_iou_sum") / max(values["images"], 1)
        summary["splits"].setdefault(split, {})[prompt] = values
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))

    by_split = defaultdict(list)
    for row in rows:
        by_split[row["split"]].append(row)
    sheet_root = args.output / "contact_sheets"
    sheet_root.mkdir(parents=True, exist_ok=True)
    for split, selected in by_split.items():
        for page, start in enumerate(range(0, len(selected), args.sheet_size)):
            chunk = selected[start:start + args.sheet_size]
            thumb_w, thumb_h, cols = 420, 120, 4
            rows_count = (len(chunk) + cols - 1) // cols
            canvas = Image.new("RGB", (cols * thumb_w, rows_count * (thumb_h + 28)), "white")
            draw = ImageDraw.Draw(canvas)
            for index, row in enumerate(chunk):
                image = Image.open(row["visualization"]).convert("RGB")
                image.thumbnail((thumb_w, thumb_h), Image.Resampling.LANCZOS)
                x, y = (index % cols) * thumb_w, (index // cols) * (thumb_h + 28)
                canvas.paste(image, (x, y))
                draw.text((x + 3, y + thumb_h + 3), Path(row["relative_path"]).name, fill="black", font=ImageFont.load_default(size=12))
            canvas.save(sheet_root / f"{split}_{page:02d}.jpg", quality=90)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
