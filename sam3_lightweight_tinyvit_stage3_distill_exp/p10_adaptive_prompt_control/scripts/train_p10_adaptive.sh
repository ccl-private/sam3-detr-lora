#!/usr/bin/env bash
set -euo pipefail
cd /slow_disk/ccl/codes/sam3
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}" \
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
.venv/bin/python -u sam3_lightweight_tinyvit_stage3_distill_exp/p10_adaptive_prompt_control/train_p10_adaptive.py \
  --data-yaml sam3_lightweight_tinyvit_stage3_distill_exp/p10_adaptive_prompt_control/configs/roadline_no_generic_negatives.yaml \
  --cache-root sam3_lightweight_tinyvit_stage3_distill_exp/p9_fresh_p8_new_teacher/cache/new_teacher_outputs \
  --feature-cache-root sam3_lightweight_tinyvit_stage3_distill_exp/cache/p1_image_features \
  --checkpoint sam3_lightweight_stage3_exp/input/efficientsam3_tinyvit_stage3.pt \
  --resume-weights sam3_lightweight_tinyvit_stage3_distill_exp/weights/p9_fresh_p8_new_teacher.best.pt \
  --save sam3_lightweight_tinyvit_stage3_distill_exp/weights/p10_adaptive_prompt_control.pt \
  --best-save sam3_lightweight_tinyvit_stage3_distill_exp/weights/p10_adaptive_prompt_control.best.pt \
  --epochs 10 --batch-size 4 --devices 4 --num-workers 8 \
  --lora-lr 1e-5 --image-lora-lr 2e-6 --head-lr 4e-6 \
  --branch-lr 2e-5 --gate-lr 2e-4 --kd-warmup-ratio 0.0 \
  --image-lora-stages 1 2 3 --save-every-epoch \
  --target-dashed-solid-instance-ratio 2.0 \
  --prompt-min-rate 0.4 --prompt-ema-new-weight 0.3 \
  --prompt-update-weight 0.2 --prompt-max-epoch-change 0.1 \
  --prompt-deadband 0.02 --prompt-performance-gain 2.0 \
  --validation-confidence-threshold 0.5 \
  --early-stop-patience 5 --early-stop-min-delta 0.002 \
  --log-name p10_adaptive_prompt_control
