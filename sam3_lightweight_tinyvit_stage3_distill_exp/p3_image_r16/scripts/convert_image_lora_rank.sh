#!/usr/bin/env bash
set -euo pipefail

cd /slow_disk/ccl/codes/sam3
./.venv/bin/python -u \
  sam3_lightweight_tinyvit_stage3_distill_exp/p3_image_r16/convert_image_lora_rank.py \
  --input sam3_lightweight_tinyvit_stage3_distill_exp/weights/p2_image_stage123_r8.best.pt \
  --checkpoint sam3_lightweight_stage3_exp/input/efficientsam3_tinyvit_stage3.pt \
  --output sam3_lightweight_tinyvit_stage3_distill_exp/weights/p3_image_r16_init.pt \
  --rank 16 \
  --alpha 32 \
  --stages 1 2 3
