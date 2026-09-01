#!/usr/bin/env bash
set -euo pipefail

cd /slow_disk/ccl/codes/sam3
experiment_dir="sam3_lightweight_tinyvit_stage3_distill_exp/p12_query_set_distill"
teacher="sam3_detr_exp/negative_prompt_ablation/weights/roadline_r8_a16_lr2e4_no_generic_negatives.best.pt"
mkdir -p "${experiment_dir}/cache/logs"

pids=()
for gpu in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES="${gpu}" .venv/bin/python -u sam3_lightweight_stage3_distill_exp/cache_teacher.py \
    --data-yaml "${experiment_dir}/configs/roadline_no_generic_negatives.yaml" \
    --teacher-lora "${teacher}" --cache-root "${experiment_dir}/cache/dense_teacher_outputs" \
    --exclude-generic-prompts --include-dense-queries --num-shards 4 --shard-index "${gpu}" \
    --image-batch-size 8 --prompt-batch-size 14 --num-workers 8 \
    > "${experiment_dir}/cache/logs/dense_teacher_gpu${gpu}.log" 2>&1 &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "${pid}"; done
