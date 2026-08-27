#!/usr/bin/env bash
set -euo pipefail

cd /slow_disk/ccl/codes/sam3

experiment_dir="sam3_lightweight_tinyvit_stage3_distill_exp/p9_fresh_p8_new_teacher"
teacher="sam3_detr_exp/negative_prompt_ablation/weights/roadline_r8_a16_lr2e4_no_generic_negatives.best.pt"
mkdir -p "${experiment_dir}/cache/logs"

pids=()
for gpu in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES="${gpu}" .venv/bin/python -u \
    sam3_lightweight_stage3_distill_exp/cache_teacher.py \
    --data-yaml "${experiment_dir}/configs/roadline_no_generic_negatives.yaml" \
    --teacher-lora "${teacher}" \
    --cache-root "${experiment_dir}/cache/new_teacher_outputs" \
    --exclude-generic-prompts --num-shards 4 --shard-index "${gpu}" \
    > "${experiment_dir}/cache/logs/new_teacher_gpu${gpu}.log" 2>&1 &
  pids+=("$!")
done
for pid in "${pids[@]}"; do
  wait "${pid}"
done
