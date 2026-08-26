#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
experiment_dir="${project_root}/sam3_detr_exp/negative_prompt_ablation"

cd "${project_root}"
mkdir -p "${experiment_dir}/logs" "${experiment_dir}/weights"

exec "${project_root}/.venv/bin/python" -u sam3_detr_exp/train_detr_lora.py \
  --data-yaml "${experiment_dir}/configs/no_generic_negatives.yaml" \
  --num-generic-negatives 0 \
  --prompt-mode class_name \
  --loss-mode sam3 \
  --train-dot-score \
  --train-seg-head \
  --accelerator gpu \
  --devices 4 \
  --precision bf16-mixed \
  --resolution 1008 \
  --batch-size 2 \
  --num-workers 8 \
  --lr 2e-4 \
  --weight-decay 1e-2 \
  --lora-rank 8 \
  --lora-alpha 16 \
  --lora-dropout 0.05 \
  --epochs 20 \
  --log-every 10 \
  --log-dir "${experiment_dir}/logs" \
  --save "${experiment_dir}/weights/roadline_r8_a16_lr2e4_no_generic_negatives.pt"

