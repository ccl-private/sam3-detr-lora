#!/usr/bin/env bash
set -euo pipefail

cd /slow_disk/ccl/codes/sam3
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}" \
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
./.venv/bin/python -u \
  sam3_lightweight_tinyvit_stage3_distill_exp/p1_image_feature/train_p1_image_feature.py \
  --data-yaml sam3_lightweight_stage3_exp/configs/roadline_lora.yaml \
  --cache-root sam3_lightweight_stage3_distill_exp/cache/p0_teacher \
  --feature-cache-root sam3_lightweight_tinyvit_stage3_distill_exp/cache/p1_image_features \
  --student-lora sam3_lightweight_tinyvit_stage3_distill_exp/weights/p3_all_r16_init.pt \
  --checkpoint sam3_lightweight_stage3_exp/input/efficientsam3_tinyvit_stage3.pt \
  --save sam3_lightweight_tinyvit_stage3_distill_exp/weights/p3_all_r16.pt \
  --image-lora-stages 1 2 3 \
  --image-lora-rank 16 \
  --image-lora-alpha 32 \
  --lora-rank 16 \
  --lora-alpha 32 \
  --log-name p3_all_r16 \
  --epochs 10 \
  --batch-size 4 \
  --num-workers 8 \
  --lora-lr 5e-5 \
  --image-lora-lr 1e-5 \
  --head-lr 2e-5 \
  --kd-weight 1.0 \
  --image-feature-kd-weight 1.0 \
  --foreground-weight 4.0 \
  --temperature 2.0 \
  --quality-threshold 0.2 \
  --devices 4 \
  --precision bf16-mixed
