#!/usr/bin/env bash
set -euo pipefail

cd /slow_disk/ccl/codes/sam3
experiment_dir="sam3_lightweight_tinyvit_stage3_distill_exp/p13_unlabeled_output_distill"
manifest="${UNLABELED_MANIFEST:-${experiment_dir}/manifests/unlabeled_train_candidates.jsonl}"
if [[ ! -f "${manifest}" ]]; then
  echo "缺少无标签训练清单：${manifest}；请先运行 prepare_unlabeled_manifests.sh。" >&2
  exit 2
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}" \
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
.venv/bin/python -u "${experiment_dir}/train_p13a_unlabeled.py" \
  --data-yaml sam3_lightweight_tinyvit_stage3_distill_exp/p12_query_set_distill/configs/roadline_no_generic_negatives.yaml \
  --cache-root sam3_lightweight_tinyvit_stage3_distill_exp/p12_query_set_distill/cache/dense_teacher_outputs \
  --feature-cache-root sam3_lightweight_tinyvit_stage3_distill_exp/cache/p1_image_features \
  --unlabeled-manifest "${manifest}" \
  --unlabeled-cache-root "${experiment_dir}/cache/unlabeled_teacher_outputs" \
  --checkpoint sam3_lightweight_stage3_exp/input/efficientsam3_tinyvit_stage3.pt \
  --save sam3_lightweight_tinyvit_stage3_distill_exp/weights/p13a_unlabeled_output_distill.pt \
  --epochs 20 --batch-size 4 --unlabeled-batch-size 4 --devices 4 --num-workers 8 \
  --image-lora-stages 1 2 3 --unlabeled-kd-weight 0.5 \
  --unlabeled-kd-warmup-ratio 0.10 --save-every-epoch \
  --log-name p13a_unlabeled_output_distill
