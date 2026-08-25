#!/usr/bin/env bash
set -euo pipefail

cd /slow_disk/ccl/codes/sam3
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}" \
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
./.venv/bin/python -u \
  sam3_lightweight_tinyvit_stage3_distill_exp/p5_dsconv_thin_line/train_p5_dsconv.py \
  --data-yaml sam3_lightweight_stage3_exp/configs/roadline_lora.yaml \
  --cache-root sam3_lightweight_stage3_distill_exp/cache/p0_teacher \
  --feature-cache-root sam3_lightweight_tinyvit_stage3_distill_exp/cache/p1_image_features \
  --student-lora sam3_lightweight_tinyvit_stage3_distill_exp/weights/p2_image_stage123_r8.best.pt \
  --checkpoint sam3_lightweight_stage3_exp/input/efficientsam3_tinyvit_stage3.pt \
  --save sam3_lightweight_tinyvit_stage3_distill_exp/weights/p5a_dsconv_frozen_20ep.pt \
  --image-lora-stages 1 2 3 \
  --log-name p5a_dsconv_frozen_20ep \
  --epochs 20 \
  --batch-size 4 \
  --num-workers 8 \
  --branch-lr 1e-4 \
  --gate-lr 1e-3 \
  --branch-channels 128 \
  --dsconv-kernel-size 9 \
  --offset-scale 1.0 \
  --kd-weight 1.0 \
  --image-feature-kd-weight 1.0 \
  --foreground-weight 4.0 \
  --temperature 2.0 \
  --quality-threshold 0.2 \
  --devices 4 \
  --precision bf16-mixed \
  --save-every-epoch
