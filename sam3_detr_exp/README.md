# SAM3 Base模块化与DETR LoRA实验

本目录负责SAM3 Base模型的模块化拆分、重组一致性验证、DETR提示推理和道路标线LoRA训练。该实验的最佳道路标线权重同时作为轻量化蒸馏教师与效果上限。

所有DETR实验代码、配置、权重、日志和输出均应放在本目录。根目录README只保留总导航和跨实验结论。

## 主要内容

- `run_video_det_modular.py`：把原始`sam3.pt`导出为10个独立模块。
- `modular_pipeline.py`：重新组装detector、tracker和video predictor。
- `run_detr_prompt_inference.py`：运行文本提示或框提示推理，可加载LoRA。
- `compare_image_original_vs_modular.py`：验证原始模型与模块化图像推理一致性。
- `compare_video_original_vs_modular.py`：验证原始模型与模块化视频推理一致性。
- `train_detr_lora.py`：道路标线DETR LoRA训练入口。
- `configs/roadline_lora.yaml`：道路标线类别和训练配置。
- `weights_lora/roadline_r8_a16_lr2e4.best.pt`：当前用于统一对比和蒸馏的教师权重。

## 模块化权重

从仓库根目录的`sam3.pt`导出：

```bash
source .venv/bin/activate
python sam3_detr_exp/run_video_det_modular.py
```

导出的10个模块包括视觉骨干、文本编码器、Transformer Encoder/Decoder、分割头、几何编码器、点积分类头，以及跟踪器的三个模块。完整模块输入输出和数据流见[模块化权重说明](docs/modular-weights.md)。

## 提示推理

```bash
python sam3_detr_exp/run_detr_prompt_inference.py \
  --image assets/images/test_image.jpg \
  --text "linear crack" \
  --lora sam3_detr_exp/weights_lora/roadline_r8_a16_lr2e4.best.pt \
  --output sam3_detr_exp/outputs/detr_text_prompt_lora.png
```

文本提示和框提示二选一；框使用原图像素坐标。视频与模块化一致性测试方法见[模块化权重说明](docs/modular-weights.md)。

## DETR LoRA训练

训练数据采用YOLO segmentation格式，类别文本来自数据集`data.yaml`。

```bash
python sam3_detr_exp/train_detr_lora.py \
  --dataset-root /slow_disk/ccl/data/crack_segment \
  --train-split train \
  --val-split val \
  --batch-size 20 \
  --epochs 20
```

LoRA挂载范围、冻结策略、损失和保存加载方式见[DETR LoRA微调说明](docs/detr-lora-finetune.md)，正式道路标线训练参数见[训练命令](docs/train-detr-lora-command.md)。

## 测试项目与结果

当前道路标线最佳权重从原始SAM3 Base模块化权重开始训练，在DETR Encoder/Decoder挂载r8 LoRA，并联合训练点积分类头和分割头。`lightning_logs/version_18/metrics.csv`显示完成了epoch 0～7共8轮完整验证，epoch 8只有训练记录、没有完成验证，因此不计作完整一轮。最低验证损失为epoch 4的4.3160。现有日志与文档没有记录提前结束的具体原因，不能把它描述为自动早停。

| 测试项目 | 结果 |
|---|---|
| 原始模型与模块化模型一致性 | 已完成图像和视频链路验证 |
| 文本提示与框提示推理 | 均已实现，可选加载LoRA |
| 道路标线DETR LoRA | 已完成，最佳权重为`roadline_r8_a16_lr2e4.best.pt` |
| 固定10图白实线评测，阈值0.5 | IoU 0.7235，Recall 0.7883 |
| 固定10图白虚线评测，阈值0.5 | IoU 0.6808，Recall 0.8226 |
| 蒸馏教师 | 已被EfficientViT和TinyViT Stage-3蒸馏实验复用 |

当前结论：Base + DETR LoRA是五条实验线中的道路标线效果上限。轻量模型是否有效，应优先在同一10图、同一提示和同一阈值下与本权重比较。由于本次训练未跑满命令中的20轮，后续引用结果时应同时注明“完成8轮验证、最佳epoch 4”，不能只写计划训练轮数。

## 目录文档

- [模块化权重与完整数据流](docs/modular-weights.md)
- [DETR LoRA微调实现](docs/detr-lora-finetune.md)
- [正式训练命令](docs/train-detr-lora-command.md)
- [多提示负样本训练待办](docs/multi-prompt-negative-training-todo.md)
