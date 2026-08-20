#!/usr/bin/env bash
set -euo pipefail

cd /slow_disk/ccl/codes/sam3

pids=()
for shard in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES="${shard}" ./.venv/bin/python -u \
    sam3_lightweight_tinyvit_stage3_distill_exp/p1_image_feature/cache_teacher_features.py \
    --data-yaml sam3_lightweight_stage3_exp/configs/roadline_lora.yaml \
    --cache-root sam3_lightweight_tinyvit_stage3_distill_exp/cache/p1_image_features \
    --resolution 1008 \
    --pool-factor 4 \
    --batch-size 4 \
    --num-workers 8 \
    --splits train val \
    --num-shards 4 \
    --shard-index "${shard}" \
    > "sam3_lightweight_tinyvit_stage3_distill_exp/logs/p1_image_feature_cache_gpu${shard}.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "${pid}" || status=$?
done
exit "${status}"
