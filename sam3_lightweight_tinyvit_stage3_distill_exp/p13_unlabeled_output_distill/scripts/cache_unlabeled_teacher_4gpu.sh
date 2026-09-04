#!/usr/bin/env bash
set -euo pipefail

cd /slow_disk/ccl/codes/sam3
experiment_dir="sam3_lightweight_tinyvit_stage3_distill_exp/p13_unlabeled_output_distill"
manifest="${UNLABELED_MANIFEST:-${experiment_dir}/manifests/unlabeled_train_candidates.jsonl}"
teacher="sam3_detr_exp/negative_prompt_ablation/weights/roadline_r8_a16_lr2e4_no_generic_negatives.best.pt"
if [[ ! -f "${manifest}" ]]; then
  echo "缺少无标签训练清单：${manifest}；请先运行 prepare_unlabeled_manifests.sh。" >&2
  exit 2
fi
mkdir -p "${experiment_dir}/cache/logs"

pids=()
for gpu in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES="${gpu}" .venv/bin/python -u "${experiment_dir}/cache_unlabeled_teacher.py" \
    --manifest "${manifest}" \
    --data-yaml "sam3_lightweight_tinyvit_stage3_distill_exp/p12_query_set_distill/configs/roadline_no_generic_negatives.yaml" \
    --teacher-lora "${teacher}" \
    --cache-root "${experiment_dir}/cache/unlabeled_teacher_outputs" \
    --threshold 0.5 --topk 50 --mask-nms-iou 0.85 --num-shards 4 --shard-index "${gpu}" \
    --image-batch-size 2 --num-workers 4 \
    > "${experiment_dir}/cache/logs/unlabeled_teacher_gpu${gpu}.log" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then status=1; fi
done
if [[ "${status}" -ne 0 ]]; then
  echo "至少一个缓存分片失败，请检查 ${experiment_dir}/cache/logs/" >&2
  exit "${status}"
fi
echo "P13-A无标签教师缓存完成。"
