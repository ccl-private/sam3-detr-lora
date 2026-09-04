#!/usr/bin/env bash
set -euo pipefail

cd /slow_disk/ccl/codes/sam3
experiment_dir="sam3_lightweight_tinyvit_stage3_distill_exp/p13_unlabeled_output_distill"
.venv/bin/python "${experiment_dir}/prepare_unlabeled_manifest.py" \
  --output-dir "${experiment_dir}/manifests" --eval-count 100

echo "P13-A默认使用 manifests/unlabeled_train_candidates.jsonl；如完成人工清理，可通过UNLABELED_MANIFEST覆盖。"
