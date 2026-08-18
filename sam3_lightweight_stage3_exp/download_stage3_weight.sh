#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
output_dir="${script_dir}/input"
output_file="${output_dir}/efficientsam3_efficientvit_stage3.pt"
partial_file="${output_file}.part"
expected_sha256="086b04b2e7da7cc98aa4621b70c7291608aa9d187357b98d03bd4d6533ed5a17"
repository_path="Simon7108528/EfficientSAM3/resolve/main/efficientsam3_ft/efficientsam3_efficientvit.pt"

mkdir -p "${output_dir}"

verify_weight() {
    local file="$1"
    [[ -f "${file}" ]] || return 1
    [[ "$(sha256sum "${file}" | awk '{print $1}')" == "${expected_sha256}" ]]
}

if verify_weight "${output_file}"; then
    echo "权重已存在且校验通过：${output_file}"
    exit 0
fi

if [[ -f "${output_file}" ]]; then
    echo "现有权重校验失败，将重新下载：${output_file}" >&2
fi

download_urls=(
    "https://hf-mirror.com/${repository_path}"
    "https://huggingface.co/${repository_path}"
)

download_succeeded=0
for url in "${download_urls[@]}"; do
    echo "正在下载：${url}"
    if curl --fail --location --retry 5 --retry-delay 2 \
        --continue-at - --output "${partial_file}" "${url}"; then
        if verify_weight "${partial_file}"; then
            mv "${partial_file}" "${output_file}"
            download_succeeded=1
            break
        fi
        echo "下载文件的 SHA-256 不正确，尝试下一个地址。" >&2
        rm -f "${partial_file}"
    else
        echo "当前地址下载失败，尝试下一个地址。" >&2
    fi
done

if [[ "${download_succeeded}" -ne 1 ]]; then
    echo "权重下载失败，请检查网络后重新运行本脚本。" >&2
    exit 1
fi

echo "下载完成：${output_file}"
echo "SHA-256：${expected_sha256}"
