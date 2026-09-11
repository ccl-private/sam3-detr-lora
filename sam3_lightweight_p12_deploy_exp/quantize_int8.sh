#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
exp=sam3_lightweight_p12_deploy_exp
output="$exp/weights/p12_white_solid_trt_int8_conservative.onnx"
if [[ -e "$output" ]]; then
    echo "Refusing to overwrite $output" >&2
    exit 1
fi
.venv/bin/python -m modelopt.onnx.quantization \
    --onnx_path "$exp/weights/p12_white_solid_trt_fp32.onnx" \
    --quantize_mode int8 --calibration_method entropy \
    --calibration_data_path "$exp/tests/output/onnx_int8/calibration_train32.npy" \
    --calibration_eps cuda:0 cpu --op_types_to_quantize Conv MatMul \
    --nodes_to_exclude '.*p5_thin_line_branch.*' '.*p6_stage1_thin_line_branch.*' \
        '.*p8_input_line_branch.*' '.*segmentation_head.*' '.*dot_prod_scoring.*' \
    --high_precision_dtype fp16 --output_path "$output" \
    --log_level INFO --log_file "$exp/tests/output/onnx_int8/quantize.log"
