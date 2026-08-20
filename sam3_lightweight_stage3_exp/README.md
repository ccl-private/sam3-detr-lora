# EfficientSAM3 Stage-3 EV-M 实验

本目录集中保存官方 Stage-3 EV-M 权重，以及用于评测该模型的本地代码和输出。模型采用 EfficientViT-B1 视觉编码器与 MobileCLIP-S0 文本编码器（上下文长度为 16），并针对文本提示概念分割进行了联合微调。

## 实验目录约束

所有 Lightweight Stage-3 相关实验都必须在当前目录 `sam3_lightweight_stage3_exp/` 内完成。新增或修改的实验代码、启动脚本、配置文件、输入数据、导出模型、评测指标和可视化结果都只能放在本目录及其子目录中。

实验过程中不得修改 `/slow_disk/ccl/codes/sam3` 主工程代码、相邻的 `/slow_disk/ccl/codes/efficientsam3` 源码，也不得修改 `sam3_lightweight_exp/`、`sam3_detr_exp/` 等其他实验目录。若需要调整上游实现，应先将相关代码复制或封装到本目录，再进行修改，确保 Stage-3 实验与其他工程完全隔离。

允许只读引用相邻 EfficientSAM3 仓库的源码和虚拟环境；这种引用不得在相邻仓库产生或覆盖任何文件。

## 下载 Stage-3 权重

本实验使用作者发布的 Stage-3 EV-M 全模型权重。作者仓库中的原始文件名是 `efficientsam3_efficientvit.pt`，位于 Hugging Face 仓库：

```text
Simon7108528/EfficientSAM3/efficientsam3_ft/efficientsam3_efficientvit.pt
```

为了明确区分早期阶段的同类权重，下载脚本会将它保存为：

```text
input/efficientsam3_efficientvit_stage3.pt
```

在本目录执行：

```bash
bash download_stage3_weight.sh
```

脚本默认先使用国内镜像 `hf-mirror.com`，镜像失败时自动回退到 Hugging Face 官方地址；它支持断点续传，并在完成后校验 SHA-256。已验证文件信息如下：

```text
文件大小：468394477 字节
SHA-256：086b04b2e7da7cc98aa4621b70c7291608aa9d187357b98d03bd4d6533ed5a17
```

也可以直接通过国内镜像下载：

```bash
curl -fL --retry 5 \
  -o input/efficientsam3_efficientvit_stage3.pt \
  'https://hf-mirror.com/Simon7108528/EfficientSAM3/resolve/main/efficientsam3_ft/efficientsam3_efficientvit.pt'
```

注意：`stage3` 是本实验为了便于识别而加入的本地文件名后缀；它和作者发布的 `efficientsam3_efficientvit.pt` 是同一份模型权重。

## 人物分割测试

在本目录运行：

```bash
bash tests/run_person.sh /path/to/image.jpg tests/output/person.png
```

不传参数时，默认使用 `tests/input/sample_person.png`。指标写入 `tests/output/person_stage3_metrics.json`。默认置信度阈值为 0.5；在随附样例上，该阈值会过滤一个分数为 0.408 的重复检测。

启动脚本使用相邻仓库 `/slow_disk/ccl/codes/efficientsam3` 中更新后的 EfficientSAM3 源码和虚拟环境。如果该仓库位置发生变化，可通过环境变量 `EFFICIENTSAM3_REPO` 覆盖路径。

## 当前样例结果

- 检测数量：2
- 置信度：0.676、0.797
- 单张 NVIDIA A800 预热后的端到端延迟：约 60 毫秒
- CUDA 峰值分配显存：约 1.10 GiB
- 运行时参数量：97.44M

## 道路标线对比测试

> **查看可视化时，只有绿色区域表示预测正确；红色表示误检，蓝色表示漏检。红色和蓝色都属于错误。**

已使用相同的 10 张道路标线图片、YAML 中全部 7 类文本提示和 0.5 置信度阈值，对 Stage-3 EV-M、Stage-1 EV-M 与 SAM3 Base 进行了像素级对比。

白色实线测试中，Stage-3 的微平均 IoU 为 0.1636，接近 Base 的 0.1760，并且精确率更高、重复候选显著更少；Stage-1 在该阈值下没有检出。三种模型对白色虚线的效果都较差，其中 Stage-3 和 Stage-1 没有超过阈值的结果。本批图片只有白色实线和白色虚线真值，其余 5 类暂时只能观察误检响应，不能据此评价正样本分割能力。

完整指标、分析和复现命令见 [tests/ROADLINE_RESULTS.md](tests/ROADLINE_RESULTS.md)。原始结果保存在 `tests/output/roadline_comparison/`。

## Stage-3 道路标线 LoRA 微调

本目录已经迁移 `sam3_lightweight_exp/` 的道路标线 LoRA 训练方案，并将基模替换为作者发布的 Stage-3 EV-M。训练时保持 EfficientViT-B1 图像编码器和 MobileCLIP-S0 文本编码器冻结，在 DETR Transformer 的注意力层与前馈层挂载 LoRA；可通过参数额外训练点积打分头和分割头。

正式训练从官方Stage-3 EV-M开始，完成计划的20轮（epoch 0～19），没有提前停止。`lightning_logs/lightning_logs/version_2/metrics.csv`中的最低验证损失为epoch 10的7.1777。使用最佳权重进行统一10图、阈值0.5复测后，白实线IoU/Recall为0.3905/0.5137，白虚线为0.2153/0.2403。结论是：完整20轮DETR LoRA训练明显改善了领域响应，但仍显著落后Base + DETR LoRA，尤其是白虚线召回。

默认采用实时文本模式：每个训练批次直接调用 Stage-3 自带的 MobileCLIP-S0 提取提示词特征。当前无需提前导出固定词表，也不会替换作者的轻量化文本编码器。

### 单卡冒烟测试

```bash
CUDA_VISIBLE_DEVICES=0 MPLCONFIGDIR=/tmp/sam3_stage3_matplotlib \
  ./.venv/bin/python -u sam3_lightweight_stage3_exp/train_lora.py \
  --max-train-samples 8 \
  --max-val-samples 4 \
  --batch-size 1 \
  --num-workers 0 \
  --train-dot-score \
  --train-seg-head \
  --dry-run
```

### 四卡正式训练

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 MPLCONFIGDIR=/tmp/sam3_stage3_matplotlib \
  ./.venv/bin/python -u sam3_lightweight_stage3_exp/train_lora.py \
  --data-yaml sam3_lightweight_stage3_exp/configs/roadline_lora.yaml \
  --text-mode runtime \
  --loss-mode sam3 \
  --train-dot-score \
  --train-seg-head \
  --accelerator gpu \
  --devices 4 \
  --precision bf16-mixed \
  --resolution 1008 \
  --batch-size 2 \
  --num-workers 8 \
  --lr 2e-4 \
  --weight-decay 1e-2 \
  --lora-rank 8 \
  --lora-alpha 16 \
  --lora-dropout 0.05 \
  --epochs 20 \
  --save sam3_lightweight_stage3_exp/weights_lora/roadline_stage3_ev_m.pt
```

默认保存两个文件：

```text
weights_lora/roadline_stage3_ev_m.best.pt  验证损失最低的权重
weights_lora/roadline_stage3_ev_m.pt       最后一轮权重
```

LoRA 权重元数据会记录 Stage-3 基模路径、文本特征模式、LoRA 参数以及是否训练打分头/分割头，避免后续加载错误基模。

### 预提取文本特征接口

当前推荐继续使用 `--text-mode runtime`。如果后续提示词完全固定，可以先把 YAML 中的 7 个类别和通用负提示词预提取出来：

```bash
CUDA_VISIBLE_DEVICES=0 ./.venv/bin/python \
  sam3_lightweight_stage3_exp/precompute_text_features.py \
  --data-yaml sam3_lightweight_stage3_exp/configs/roadline_lora.yaml \
  --output sam3_lightweight_stage3_exp/text_features/roadline_mobileclip_s0.pt
```

训练时只需切换两个参数，其他代码和损失流程保持不变：

```bash
./.venv/bin/python -u sam3_lightweight_stage3_exp/train_lora.py \
  --text-mode precomputed \
  --text-cache sam3_lightweight_stage3_exp/text_features/roadline_mobileclip_s0.pt \
  --train-dot-score \
  --train-seg-head
```

预提取模式遇到缓存中不存在的提示词会立即报错。修改 YAML 类别名或负提示词后必须重新执行预提取脚本。缓存文件保留特征来源权重、编码器类型、上下文长度和 YAML 路径等元数据。

### 已完成的验证

使用 2 张训练图片、2 张验证图片各运行一个批次：

| 文本模式 | 总参数量 | 可训练参数量 | 训练损失 | 验证损失 | 状态 |
|---|---:|---:|---:|---:|---|
| 实时 MobileCLIP-S0 | 98.25M | 4.29M | 135.903 | 25.909 | 通过 |
| 预提取文本特征 | 55.70M | 4.29M | 135.832 | 25.883 | 通过 |

`weights_lora/smoke_runtime*.pt` 和 `weights_lora/smoke_precomputed*.pt` 只是单批次链路验证权重，不能用于正式效果评估。

## 目录结构

```text
configs/          LoRA 训练配置
input/            官方 Stage-3 基模权重
lightning_logs/   LoRA 训练日志
text_features/    可选的预提取文本特征
weights_lora/     LoRA 训练权重
tests/            所有训练无关的测试代码、样例、结果和文档
```

训练相关文件：

```text
download_stage3_weight.sh  Stage-3 EV-M 权重下载及校验脚本
train_lora.py              Stage-3 道路标线 LoRA 训练入口
model_adapter.py           Stage-3 模型构建、冻结与 LoRA 挂载
text_feature_provider.py   实时/预提取文本特征兼容接口
precompute_text_features.py  固定提示词特征预提取脚本
configs/roadline_lora.yaml 道路标线数据与提示词配置
```

测试相关文件：

```text
tests/run_person.sh         person 文本提示测试入口
tests/test_stage3_person.py 单图推理与指标记录脚本
tests/benchmark_roadline.py Stage-3、Stage-1 与 Base 道路标线对比脚本
tests/ROADLINE_RESULTS.md   道路标线测试结果与分析
tests/input/                测试样例
tests/output/               测试指标与可视化结果
```
