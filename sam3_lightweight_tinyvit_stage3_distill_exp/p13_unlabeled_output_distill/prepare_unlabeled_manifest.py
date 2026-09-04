#!/usr/bin/env python3
"""从现有教师抽检结果生成P13候选清单；正式训练前仍需人工清理。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def stable_order(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("/slow_disk/ccl/data/roadline_unlabeled_distill"))
    parser.add_argument("--comparison-root", type=Path, default=Path("/slow_disk/ccl/data/roadline_unlabeled_distill/teacher_comparison/threshold_05/parts"))
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "manifests")
    parser.add_argument("--eval-count", type=int, default=100)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    by_path = {}
    for part in sorted(args.comparison_root.glob("part_*.jsonl")):
        for raw in part.read_text().splitlines():
            if not raw.strip():
                continue
            row = json.loads(raw)
            positives = [item["prompt"] for item in row["prompts"] if int(item["new_teacher"]["detections"]) > 0]
            if not positives:
                continue
            relative = row["relative_path"]
            if not (args.data_root / relative).exists():
                raise FileNotFoundError(f"教师记录对应图片不存在：{args.data_root / relative}")
            by_path[relative] = {
                "relative_path": relative,
                "category": row.get("category"),
                "platform": row.get("platform"),
                "teacher_positive_prompts_at_0_5": positives,
            }
    rows = sorted(by_path.values(), key=lambda row: stable_order(row["relative_path"]))
    if args.limit is not None:
        rows = rows[:args.limit]
    eval_count = min(max(args.eval_count, 0), len(rows))
    eval_rows, train_rows = rows[:eval_count], rows[eval_count:]
    write_jsonl(args.output_dir / "unlabeled_all_candidates.jsonl", rows)
    write_jsonl(args.output_dir / "unlabeled_eval_candidates.jsonl", eval_rows)
    write_jsonl(args.output_dir / "unlabeled_train_candidates.jsonl", train_rows)
    print(f"新教师阈值0.5候选={len(rows)}，预留评测={len(eval_rows)}，训练候选={len(train_rows)}")
    print("注意：这些只是自动候选清单。正式训练前必须人工删除错图、重复图和教师明显误检，并另存为clean文件。")


if __name__ == "__main__":
    main()
