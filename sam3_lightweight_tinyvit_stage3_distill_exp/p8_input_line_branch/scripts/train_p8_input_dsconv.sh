#!/usr/bin/env bash
set -euo pipefail

cd /slow_disk/ccl/codes/sam3
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}" \
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
./.venv/bin/python -u \
  sam3_lightweight_tinyvit_stage3_distill_exp/p8_input_line_branch/train_p8_input_line.py \
  --data-yaml sam3_lightweight_stage3_exp/configs/roadline_lora.yaml \
  --cache-root sam3_lightweight_stage3_distill_exp/cache/p0_teacher \
  --feature-cache-root sam3_lightweight_tinyvit_stage3_distill_exp/cache/p1_image_features \
  --student-lora sam3_lightweight_tinyvit_stage3_distill_exp/weights/p7_highres_fpn_frozen_p6.epoch9.pt \
  --checkpoint sam3_lightweight_stage3_exp/input/efficientsam3_tinyvit_stage3.pt \
  --save sam3_lightweight_tinyvit_stage3_distill_exp/weights/p8_input_dsconv_frozen_p7.pt \
  --image-lora-stages 1 2 3 \
  --operator dsconv \
  --stem-channels 16 \
  --line-channels 16 \
  --kernel-size 9 \
  --offset-scale 1.0 \
  --log-name p8_input_dsconv_frozen_p7 \
  --epochs 5 \
  --batch-size 4 \
  --num-workers 8 \
  --branch-lr 1e-4 \
  --gate-lr 1e-3 \
  --kd-weight 1.0 \
  --image-feature-kd-weight 1.0 \
  --foreground-weight 4.0 \
  --temperature 2.0 \
  --quality-threshold 0.2 \
  --devices 4 \
  --precision bf16-mixed \
  --save-every-epoch
