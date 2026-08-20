#!/usr/bin/env bash
set -euo pipefail

cd /slow_disk/ccl/codes/sam3
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}" \
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
./.venv/bin/python -u \
  sam3_lightweight_tinyvit_stage3_distill_exp/train_p0_image_lora.py \
  --data-yaml sam3_lightweight_stage3_exp/configs/roadline_lora.yaml \
  --cache-root sam3_lightweight_stage3_distill_exp/cache/p0_teacher \
  --student-lora sam3_lightweight_stage3_exp/weights_lora/roadline_stage3_ev_m.best.pt \
  --checkpoint sam3_lightweight_stage3_exp/input/efficientsam3_tinyvit_stage3.pt \
  --save sam3_lightweight_tinyvit_stage3_distill_exp/weights/p0_image_lora_r8.pt \
  --epochs 10 \
  --batch-size 4 \
  --num-workers 8 \
  --lora-lr 5e-5 \
  --image-lora-lr 1e-5 \
  --head-lr 2e-5 \
  --kd-weight 1.0 \
  --temperature 2.0 \
  --quality-threshold 0.2 \
  --devices 4 \
  --precision bf16-mixed
