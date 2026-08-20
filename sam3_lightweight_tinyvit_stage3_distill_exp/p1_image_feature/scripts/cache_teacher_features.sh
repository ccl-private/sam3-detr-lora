#!/usr/bin/env bash
set -euo pipefail

cd /slow_disk/ccl/codes/sam3
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" ./.venv/bin/python -u \
  sam3_lightweight_tinyvit_stage3_distill_exp/p1_image_feature/cache_teacher_features.py \
  --data-yaml sam3_lightweight_stage3_exp/configs/roadline_lora.yaml \
  --cache-root sam3_lightweight_tinyvit_stage3_distill_exp/cache/p1_image_features \
  --resolution 1008 \
  --pool-factor 4 \
  --batch-size 4 \
  --num-workers 8 \
  --splits train val
