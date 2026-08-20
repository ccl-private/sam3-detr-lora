#!/usr/bin/env bash
set -euo pipefail

cd /slow_disk/ccl/codes/sam3
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" ./.venv/bin/python -u \
  sam3_lightweight_stage3_distill_exp/cache_teacher.py \
  --data-yaml sam3_lightweight_stage3_exp/configs/roadline_lora.yaml \
  --teacher-lora sam3_detr_exp/weights_lora/roadline_r8_a16_lr2e4.best.pt \
  --cache-root sam3_lightweight_stage3_distill_exp/cache/p0_teacher \
  --resolution 1008 \
  --image-batch-size 8 \
  --prompt-batch-size 144 \
  --num-workers 8 \
  --splits train val
