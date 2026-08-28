#!/usr/bin/env bash
set -euo pipefail

cd /slow_disk/ccl/codes/sam3

experiment_dir="sam3_lightweight_tinyvit_stage3_distill_exp/p9_fresh_p8_new_teacher"
weights_dir="sam3_lightweight_tinyvit_stage3_distill_exp/weights"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}" \
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
.venv/bin/python -u \
  "${experiment_dir}/train_p9_fresh_p8.py" \
  --data-yaml "${experiment_dir}/configs/roadline_no_generic_negatives.yaml" \
  --cache-root "${experiment_dir}/cache/new_teacher_outputs" \
  --feature-cache-root sam3_lightweight_tinyvit_stage3_distill_exp/cache/p1_image_features \
  --checkpoint sam3_lightweight_stage3_exp/input/efficientsam3_tinyvit_stage3.pt \
  --resume-weights "${weights_dir}/p9_fresh_p8_new_teacher.best.pt" \
  --save "${weights_dir}/p9_continue_from_epoch19.pt" \
  --best-save "${weights_dir}/p9_continue_from_epoch19.best.pt" \
  --epochs 20 --batch-size 4 --devices 4 --num-workers 8 \
  --lora-lr 1e-5 --image-lora-lr 2e-6 --head-lr 4e-6 \
  --branch-lr 2e-5 --gate-lr 2e-4 --kd-warmup-ratio 0.0 \
  --image-lora-stages 1 2 3 --save-every-epoch \
  --early-stop-monitor val/supervised \
  --early-stop-patience 5 --early-stop-min-delta 0.005 \
  --log-name p9_continue_from_epoch19
