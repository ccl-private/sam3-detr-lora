# 轻量化 EfficientSAM3 实验

本目录采用独立隔离设计。所有轻量化实验代码、配置、生成的预训练包、LoRA 权重、日志和推理输出均保存在 `sam3_lightweight_exp/` 下。外部 EfficientSAM3 仓库以只读方式提供上游模型定义和源权重。

源 TinyViT-S 权重包含较大的文本编码器。为了在支持道路标线类别的同时保留小型固定提示设计，导出步骤会预计算 YAML 中列出的全部类别提示和负提示，并用无参数的固定词表查找模块替换文本编码器。

## 1. 导出固定词表预训练模型

```bash
CUDA_VISIBLE_DEVICES=0 ./.venv/bin/python sam3_lightweight_exp/export_fixed_vocabulary.py \
  --data-yaml sam3_lightweight_exp/configs/roadline_lora.yaml
```

## 2. 训练冒烟测试

```bash
CUDA_VISIBLE_DEVICES=0 ./.venv/bin/python sam3_lightweight_exp/train_lora.py \
  --max-train-samples 8 --max-val-samples 4 --batch-size 1 --num-workers 0 \
  --train-dot-score --train-seg-head --dry-run
```

## 3. 四卡训练

```bash
mkdir -p sam3_lightweight_exp/logs
./.venv/bin/python -u sam3_lightweight_exp/train_lora.py \
  --data-yaml sam3_lightweight_exp/configs/roadline_lora.yaml \
  --loss-mode sam3 --train-dot-score --train-seg-head \
  --accelerator gpu --devices 4 --precision bf16-mixed \
  --resolution 1008 --batch-size 2 --num-workers 8 \
  --lr 2e-4 --weight-decay 1e-2 \
  --lora-rank 8 --lora-alpha 16 --lora-dropout 0.05 \
  --epochs 20 --log-every 10 \
  --save sam3_lightweight_exp/weights_lora/roadline_tinyvit_s.pt \
  2>&1 | tee sam3_lightweight_exp/logs/roadline_tinyvit_s.log
```

固定词表由 YAML 生成。新增或重命名 YAML 中的提示词后，必须重新执行导出；运行时遇到未知提示词会立即报错。

## 4. 测试前十张道路标线图片

```bash
CUDA_VISIBLE_DEVICES=0 ./.venv/bin/python sam3_lightweight_exp/infer_first10.py \
  --lora sam3_lightweight_exp/weights_lora/roadline_tinyvit_s.best.pt \
  --images /mnt/mnt108_hdd/biaozhu/labeled/shenxing/12/segment/roadline/20260106 \
  --output sam3_lightweight_exp/outputs/roadline_first10
```

初始单轮训练可行性结果见 `RESULTS.md`。
