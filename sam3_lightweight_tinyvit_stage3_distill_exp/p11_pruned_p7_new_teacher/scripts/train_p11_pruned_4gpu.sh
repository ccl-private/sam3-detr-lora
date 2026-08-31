#!/usr/bin/env bash
set -euo pipefail

cd /slow_disk/ccl/codes/sam3

experiment_dir="sam3_lightweight_tinyvit_stage3_distill_exp/p11_pruned_p7_new_teacher"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}" \
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
.venv/bin/python -u \
  "${experiment_dir}/train_p11_pruned.py" \
  --data-yaml "${experiment_dir}/configs/roadline_no_generic_negatives.yaml" \
  --cache-root sam3_lightweight_tinyvit_stage3_distill_exp/p9_fresh_p8_new_teacher/cache/new_teacher_outputs \
  --feature-cache-root sam3_lightweight_tinyvit_stage3_distill_exp/cache/p1_image_features \
  --checkpoint sam3_lightweight_stage3_exp/input/efficientsam3_tinyvit_stage3.pt \
  --save sam3_lightweight_tinyvit_stage3_distill_exp/weights/p11_pruned_p7_new_teacher.pt \
  --epochs 20 --batch-size 4 --devices 4 --num-workers 8 \
  --image-lora-stages 1 2 3 --save-every-epoch \
  --log-name p11_pruned_p7_new_teacher
