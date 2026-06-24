# sam3_detr_exp

这个目录现在只保留一条非 JIT 的模块化主线：

- 从原始 `sam3.pt` 导出模块权重
- 用 `weights_modular/*.pt` 重新组装 detector / tracker
- 对比原始模型和模块化模型的推理结果
- 单独跑 detector 提示推理

DETR LoRA 训练部分现在也已经整理成清楚的分层结构：

- `train_detr_lora.py`
  - Lightning 2.6.5 启动入口
- `model/`
  - 训练相关模型封装
- `utils/`
  - dataset / LoRA / loss / save-load 工具

如果你只想知道“这里每个文件是干什么的、怎么用”，先看这份 README。
如果你要看模块输入输出、shape、数据流图，再看 [docs/modular-weights.md](/slow_disk/ccl/codes/sam3/sam3_detr_exp/docs/modular-weights.md)。
如果你后面要做 DETR 的 LoRA 微调，再看 [docs/detr-lora-finetune.md](/slow_disk/ccl/codes/sam3/sam3_detr_exp/docs/detr-lora-finetune.md)。

## Directory Overview

### [run_video_det_modular.py](/slow_disk/ccl/codes/sam3/sam3_detr_exp/run_video_det_modular.py)

用途：

- 从原始 `sam3.pt` 拆出模块权重
- 生成 `weights_modular/*.pt`

什么时候用：

- 第一次准备模块化权重时
- 原始 checkpoint 更新后，想重新导出模块权重时

怎么用：

```bash
source /slow_disk/ccl/codes/sam3/.venv/bin/activate
python sam3_detr_exp/run_video_det_modular.py
```

输出：

- `sam3_detr_exp/weights_modular/*.pt`

### [modular_pipeline.py](/slow_disk/ccl/codes/sam3/sam3_detr_exp/modular_pipeline.py)

用途：

- 这是整个目录的核心组装入口
- 负责从 `weights_modular/*.pt` 组装：
  - detector
  - tracker
  - video model

主要接口：

- `build_detector_modules()`
- `build_detector_model()`
- `build_tracker_modules()`
- `build_tracker_model()`
- `build_video_model()`
- `ModularVideoPredictor`

什么时候用：

- 你写自己的推理脚本时
- 你后面想做模块级微调 / 蒸馏 / ONNX 包装时
- 你想单独调用某个模块做 I/O 测试时

### [compare_image_original_vs_modular.py](/slow_disk/ccl/codes/sam3/sam3_detr_exp/compare_image_original_vs_modular.py)

用途：

- 在同一张图上对比：
  - 原始 `sam3.pt`
  - 模块化 `weights_modular`

什么时候用：

- 验证模块化 detector 的结果是否和原始模型一致
- 快速肉眼对比 box / mask 是否重合

怎么用：

```bash
python sam3_detr_exp/compare_image_original_vs_modular.py \
  --image assets/images/test_image.jpg \
  --prompt shoe
```

默认输出：

- `sam3_detr_exp/outputs/image_original_vs_modular.png`

### [compare_video_original_vs_modular.py](/slow_disk/ccl/codes/sam3/sam3_detr_exp/compare_video_original_vs_modular.py)

用途：

- 在同一段视频上对比：
  - 原始 `sam3.pt`
  - 模块化 `weights_modular`

什么时候用：

- 验证模块化 video pipeline 是否和原始模型一致
- 看 tracking / id / mask 是否明显分叉

怎么用：

```bash
python sam3_detr_exp/compare_video_original_vs_modular.py \
  --video assets/videos/bedroom.mp4 \
  --prompt person \
  --max-frames 2
```

默认输出：

- `sam3_detr_exp/outputs/video_original_vs_modular.mp4`
- `sam3_detr_exp/outputs/video_original_vs_modular.png`

说明：

- `png` 是首帧预览
- `mp4` 是逐帧对比视频

### [run_detr_prompt_inference.py](/slow_disk/ccl/codes/sam3/sam3_detr_exp/run_detr_prompt_inference.py)

用途：

- 只跑 modular detector
- 支持两种提示：
  - 文本提示 `--text`
  - 框提示 `--box`

什么时候用：

- 你只想验证 DETR 那半边
- 你想单独看 detector 的目标分割结果
- 你后面想把 detector 单独抽出来时

怎么用：

文本提示：

```bash
python sam3_detr_exp/run_detr_prompt_inference.py \
  --image assets/images/test_image.jpg \
  --text shoe \
  --output sam3_detr_exp/outputs/detr_text_prompt.png
```

加载 LoRA 后推理：

```bash
python sam3_detr_exp/run_detr_prompt_inference.py \
  --image assets/images/test_image.jpg \
  --text "linear crack" \
  --lora sam3_detr_exp/weights_lora/detr_transformer_lora.pt \
  --output sam3_detr_exp/outputs/detr_text_prompt_lora.png
```

框提示：

```bash
python sam3_detr_exp/run_detr_prompt_inference.py \
  --image assets/images/test_image.jpg \
  --box 320,300,980,690 \
  --output sam3_detr_exp/outputs/detr_box_prompt.png
```

说明：

- `--box` 格式是像素坐标 `x0,y0,x1,y1`
- 脚本内部会自动转成模型需要的归一化 `cxcywh`
- `--lora` 可选，用来加载 `train_detr_lora.py` 训练出来的增量权重

### [train_detr_lora.py](/slow_disk/ccl/codes/sam3/sam3_detr_exp/train_detr_lora.py)

用途：

- Lightning 2.6.5 训练入口
- 在当前 modular detector 上做 detector-only LoRA 微调
- 直接读取 `/slow_disk/ccl/data/crack_segment` 的 YOLO segmentation 数据集

什么时候用：

- 你想先验证 LoRA 是否能挂到当前 transformer 上
- 你想先跑 detector-only 微调，不接 tracker
- 你想直接在裂缝分割数据集上训练

怎么用：

最小 dry-run：

```bash
python sam3_detr_exp/train_detr_lora.py \
  --dataset-root /slow_disk/ccl/data/crack_segment \
  --train-split train \
  --val-split val \
  --max-train-samples 1 \
  --max-val-samples 1 \
  --dry-run
```

正式训练：

```bash
python sam3_detr_exp/train_detr_lora.py \
  --dataset-root /slow_disk/ccl/data/crack_segment \
  --train-split train \
  --val-split val \
  --batch-size 20 \
  --epochs 20
```

说明：

- 当前只走文本提示训练
- 当前框架版本：
  - `lightning==2.6.5`
- 示例里使用 `--batch-size 20`
- 默认 prompt 使用类别名：
  - `linear crack`
  - `alligator crack`
  - `pothole`
- 如果你想统一都用一个词，可以加：
  - `--prompt-mode generic --generic-prompt crack`
- 输出默认保存到 `sam3_detr_exp/weights_lora/detr_transformer_lora.pt`

训练后验证：

```bash
python sam3_detr_exp/run_detr_prompt_inference.py \
  --image assets/images/test_image.jpg \
  --text "linear crack" \
  --lora sam3_detr_exp/weights_lora/detr_transformer_lora.pt \
  --output sam3_detr_exp/outputs/detr_text_prompt_lora.png
```

### [model/](/slow_disk/ccl/codes/sam3/sam3_detr_exp/model)

用途：

- 放训练相关模型封装
- 当前主要是：
  - `detr_lora_module.py`
  - `DetrLoraLightningModule`

### [utils/](/slow_disk/ccl/codes/sam3/sam3_detr_exp/utils)

用途：

- 放 DETR LoRA 训练公共工具
- 当前包括：
  - `detr_lora_data.py`
    - YOLO segmentation dataset
    - LightningDataModule
  - `detr_lora_utils.py`
    - LoRA 挂载 / save-load
    - detector 组装
    - prompt / target 构造
    - matcher + loss

### [weights_modular/](/slow_disk/ccl/codes/sam3/sam3_detr_exp/weights_modular)

用途：

- 保存模块化拆分后的 `state_dict`

里面包括：

- `vision_backbone.pt`
- `text_encoder.pt`
- `transformer_encoder.pt`
- `transformer_decoder.pt`
- `segmentation_head.pt`
- `geometry_encoder.pt`
- `dot_product_scoring.pt`
- `tracker_sam_heads.pt`
- `tracker_maskmem_backbone.pt`
- `tracker_transformer.pt`

说明：

- 这些不是可直接裸跑的计算图
- 它们必须通过 [modular_pipeline.py](/slow_disk/ccl/codes/sam3/sam3_detr_exp/modular_pipeline.py) 重新组装

### [docs/modular-weights.md](/slow_disk/ccl/codes/sam3/sam3_detr_exp/docs/modular-weights.md)

用途：

- 模块化权重和模块接口说明书
- 包含：
  - 每个模块输入输出
  - 实测 shape
  - detector / tracker 数据流图
  - 当前目录最终工作流

什么时候看：

- 想理解模块边界时
- 想做模块级微调 / 蒸馏 / ONNX 时
- 想确认每个模块实际吃什么、吐什么时

### [docs/detr-lora-finetune.md](/slow_disk/ccl/codes/sam3/sam3_detr_exp/docs/detr-lora-finetune.md)

用途：

- 说明怎么在当前非 JIT 模块化方案上做 DETR LoRA 微调
- 包含：
  - 推荐训练边界
  - 推荐冻结/训练模块
  - LoRA 挂载位置
  - loss 和数据组织建议
  - 保存 / 加载 / 部署建议

什么时候看：

- 你准备开始做 detector 微调时
- 你想先做 LoRA，再做蒸馏时
- 你想保持模块化、后续还能独立替换 detector 时

### [outputs/](/slow_disk/ccl/codes/sam3/sam3_detr_exp/outputs)

用途：

- 保存对比图、对比视频、detector 可视化结果

说明：

- 这是运行产物目录
- 已经在 `.gitignore` 里忽略，不会默认提交

## Recommended Usage

### 1. 先导出模块权重

```bash
source /slow_disk/ccl/codes/sam3/.venv/bin/activate
python sam3_detr_exp/run_video_det_modular.py
```

### 2. 验证 detector-only

```bash
python sam3_detr_exp/run_detr_prompt_inference.py \
  --image assets/images/test_image.jpg \
  --text shoe
```

### 3. 验证图片结果

```bash
python sam3_detr_exp/compare_image_original_vs_modular.py \
  --image assets/images/test_image.jpg \
  --prompt shoe
```

### 4. 验证视频结果

```bash
python sam3_detr_exp/compare_video_original_vs_modular.py \
  --video assets/videos/bedroom.mp4 \
  --prompt person \
  --max-frames 2
```

## Which File To Use

如果你现在的目标是：

- “我要重新导出模块权重”
  - 用 `run_video_det_modular.py`

- “我要写自己的 modular 推理代码”
  - 用 `modular_pipeline.py`

- “我要看原始模型和模块化模型是不是一样”
  - 用 `compare_image_original_vs_modular.py`
  - 或 `compare_video_original_vs_modular.py`

- “我只想测试 detector，从文本或框提示直接到分割结果”
  - 用 `run_detr_prompt_inference.py`

- “我想知道模块接口、shape、数据流”
  - 看 `docs/modular-weights.md`
