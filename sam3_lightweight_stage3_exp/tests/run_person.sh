#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EFFICIENTSAM3_REPO="${EFFICIENTSAM3_REPO:-$(cd "$HERE/../../../efficientsam3" && pwd)}"
IMAGE="${1:-$HERE/input/sample_person.png}"
OUTPUT="${2:-$HERE/output/person_stage3.png}"

cd "$HERE"
MPLCONFIGDIR="$HERE/.matplotlib" \
PYTHONPATH="$EFFICIENTSAM3_REPO/sam3:$HERE" \
  "$EFFICIENTSAM3_REPO/.venv/bin/python" test_stage3_person.py \
  --image "$IMAGE" \
  --output "$OUTPUT"
