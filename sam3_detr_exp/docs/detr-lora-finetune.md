# SAM3 DETR LoRA 微调方案

只需要执行训练时，直接查看独立的
[DETR LoRA 训练命令手册](train-detr-lora-command.md)。

## 环境基线

本实验目录当前以仓库根目录的 [requirements.txt](../../requirements.txt) 作为实际依赖基线。

说明：

- 这份 `requirements.txt` 直接来自当前 `.venv` 的 `pip freeze`
- 不再以原始上游 SAM3 的 `pyproject.toml` 依赖为准
- 当前验证环境是 `Python 3.13.11`
- 当前训练框架版本是 `lightning 2.6.5`

这份文档只讨论 `sam3_detr_exp` 这条非 JIT 模块化链路下，怎么对 DETR 部分做 LoRA 微调。

目标很明确：

- 不动整套 SAM3 视频系统
- 先只动 detector 这半边
- 尽量冻结大模型主体
- 让训练入口、保存格式、推理复用路径都保持清晰

当前最适合做 LoRA 的对象，不是整个 `sam3.pt`，而是模块化后的 detector 子模块。

## 数据格式要求

当前训练代码只支持文本提示的 detector-only 数据组织，使用的是 YOLO segmentation 标注。

### YAML 与目录结构

训练入口通过 `--data-yaml` 指定数据配置，不再单独传 dataset root 或 split 名。YAML 示例：

```yaml
path: /data/my_dataset
train: train
val: val

names:
  0: class one prompt
  1: class two prompt
  2: class three prompt
```

对应目录至少包含：

```text
/data/my_dataset/
  train/
    0001.jpg
    0001.txt
    0002.jpg
    0002.txt
  val/
    0101.jpg
    0101.txt
```

说明：

- `path` 支持绝对路径；相对路径以 YAML 文件所在目录为基准
- `train` 和 `val` 支持相对于 `path` 的路径，也支持绝对路径
- `train` 必填，`val` 可省略
- 当前实现假设“图片和标签在同一目录，同名不同后缀”
- 不读取 `images/train`、`labels/train` 这种分层目录

### 图片格式

当前 dataset loader 支持：

- `.jpg`
- `.jpeg`
- `.png`
- `.bmp`

读取后会统一：

- 转成 RGB
- resize 到 `--resolution`
- normalize 到当前 detector 预处理格式

### `data.yaml` 格式

当前要求 YAML 包含 `path`、`train` 和 `names`，验证时再提供 `val`：

```yaml
names:
  0: class one prompt
  1: class two prompt
  2: class three prompt
```

要求：

- key 是类别 id
- value 是类别名
- 训练时如果 `--prompt-mode class_name`，文本提示就直接来自这里
- 下划线会自动替换成空格，例如 `class_one -> class one`

### 标签格式

每个 `.txt` 文件使用 YOLO segmentation 每行一个实例：

```text
class_id x1 y1 x2 y2 x3 y3 ...
```

要求：

- 第 1 列是 `class_id`
- 后面是 polygon 顶点序列
- 坐标是相对原图的归一化坐标，范围 `[0, 1]`
- 一行至少要能形成 3 个点，所以最少 `7` 列
- 点数可以大于 3

单行示例：

```text
0 0.125 0.210 0.180 0.215 0.240 0.260 0.230 0.320
```

### 当前训练样本是怎么构造的

当前不是“每个 polygon 一条样本”，而是：

- 先按图片读取
- 再按 `class_id` 分组
- 同一张图中同一类别的多个 polygon 会合成一个训练样本

也就是说，一个 sample 大致对应：

- 一张图
- 一个文本提示
- 这个提示对应的一组 `gt_boxes`
- 这组实例对应的一组 `gt_masks`

### 当前 prompt 来源

训练时文本提示有两种模式：

- `--prompt-mode class_name`
  - 从 `data.yaml` 的类别名读取
- `--prompt-mode generic --generic-prompt object`
  - 所有样本统一使用一个固定文本提示

## 1. 推荐微调范围

推荐按下面这个优先级来做。

### 第一优先级：只给 DETR 变换器加 LoRA

对应模块：

- `transformer_encoder.pt`
- `transformer_decoder.pt`

也就是 [modular_pipeline.py](../modular_pipeline.py) 里：

- `detector.transformer.encoder`
- `detector.transformer.decoder`

这是最稳的一层，原因有三点：

- 参数规模远小于 `vision_backbone` 和 `text_encoder`
- 直接决定 query 和 image/text prompt 的融合方式
- 对检测框、mask、prompt 对齐都会有直接影响

如果你是第一次做，建议先只改这里。

### 第二优先级：允许少量预测头一起训练

可选模块：

- `dot_product_scoring.pt`
- `segmentation_head.pt`

适用情况：

- 新数据分布下分类分数偏移明显
- 框已经大致对，但 mask 边界不够贴合
- 你想让 LoRA 之外再给少量轻量头部参数自由度

建议做法：

- 先只训 LoRA
- 如果效果不够，再解冻 `dot_product_scoring`
- 最后再考虑解冻 `segmentation_head`

### 不建议一开始就动的大模块

先冻结：

- `vision_backbone.pt`
- `text_encoder.pt`
- `geometry_encoder.pt`
- `tracker_sam_heads.pt`
- `tracker_maskmem_backbone.pt`
- `tracker_transformer.pt`

原因：

- `vision_backbone` 很大，显存和训练稳定性压力都高
- `text_encoder` 很大，而且通常不是 DETR 适配的第一瓶颈
- `geometry_encoder` 更多是在框/点提示编码层，先不该成为主要训练变量
- tracker 模块和当前“提示到 DETR 分割结果”的目标无关

## 2. 建议的训练边界

先把目标收窄成一件事：

`图像 + 文本提示` 或 `图像 + 框提示` -> `DETR boxes / scores / masks`

也就是说，优先围绕 [run_detr_prompt_inference.py](../run_detr_prompt_inference.py) 这条 detector-only 链路做训练，不要一开始就接视频 tracker。

推荐分成两种任务：

### 方案 A：文本提示检测/分割

输入：

- image
- text prompt

输出监督：

- `pred_boxes`
- `pred_logits`
- `pred_masks`

适合：

- 类别词驱动的目标检出
- 开放词表或少样本类别迁移

### 方案 B：框提示细化

输入：

- image
- prompt box

输出监督：

- refined boxes
- masks

适合：

- 已有外部检测器提供候选框
- 只想把 SAM3 DETR 用作 refinement / segmentation head

如果你后面目标是“某个垂类数据集上的文本检出”，优先做方案 A。
如果你后面目标是“已有框，想抠得更准”，优先做方案 B。

## 3. 为什么模块化方案更适合 LoRA

相比直接抱着整份 `sam3.pt` 做，模块化的好处很实际：

- 训练边界清楚
- 能明确指定哪些模块可训练
- 保存出来的结果可以继续按模块管理
- 未来可以只替换 detector，不影响 tracker
- 后面做蒸馏、剪枝、ONNX、TensorRT 时更容易拆分

这也是当前不继续走 JIT 的主要原因。

JIT 更偏部署封装。
你现在这个目标更偏“可训练、可替换、可继续演化”。

## 4. 推荐的参数更新策略

最推荐的第一版：

- 冻结 `vision_backbone`
- 冻结 `text_encoder`
- 冻结 `geometry_encoder`
- 冻结 `dot_product_scoring`
- 冻结 `segmentation_head`
- 冻结全部 tracker
- 仅在 `transformer.encoder` / `transformer.decoder` 上挂 LoRA

训练若不足，再逐步放开：

1. `transformer encoder + decoder` LoRA only
2. `+ dot_product_scoring` full finetune
3. `+ segmentation_head` full finetune

不建议一上来就把 backbone 解冻。

## 5. LoRA 挂载位置建议

LoRA 通常优先挂在线性层。

在 transformer 里，优先检查这些层：

- `q_proj`
- `k_proj`
- `v_proj`
- `out_proj`
- `fc1`
- `fc2`

如果模块实现里没有显式拆成这些名字，也可以退一步，对以下线性层模式做匹配：

- attention 内部 `nn.Linear`
- MLP / FFN 内部 `nn.Linear`

推荐优先级：

1. attention 的 `q/v`
2. attention 的 `q/k/v/out`
3. 再加 FFN 的 `fc1/fc2`

如果你要节省显存和训练时间，第一版只挂 `q_proj` 和 `v_proj` 就够了。

## 6. 推荐超参数起点

先给一个保守、容易起跑的配置：

- LoRA rank: `8`
- LoRA alpha: `16`
- LoRA dropout: `0.05`
- learning rate:
  - LoRA 参数：`1e-4` 到 `3e-4`
  - 若解冻 head：`5e-5` 到 `1e-4`
- weight decay: `0.01`
- batch size:
  - 单卡显存紧张时从 `1` 或 `2` 起
- mixed precision:
  - `bf16` 优先
- gradient clip:
  - `1.0`
- warmup:
  - 前 `2%` 到 `5%` step

如果数据很少，rank 可以从 `4` 起。
如果数据分布和原始 SAM3 差异很大，可以升到 `16`。

## 7. 损失设计建议

如果沿用原 DETR 头输出，建议保留三类主损失：

- 分类损失
- box 回归损失
- mask 损失

典型组合：

- classification:
  - focal loss 或原实现中的分类损失
- box:
  - `L1 + GIoU`
- mask:
  - `BCE / focal + dice`

如果你只关心分割，依然不建议把 box loss 全去掉。
因为 DETR query 对齐通常靠 box supervision 稳得多。

## 8. 训练数据应该怎么喂

最稳妥的是先做 detector-only dataset。

样本结构建议统一成：

```python
sample = {
    "image": image,
    "prompt_type": "text" or "box",
    "text": "shoe",              # prompt_type == "text" 时使用
    "box_prompt": [cx, cy, w, h],# prompt_type == "box" 时使用，归一化坐标
    "gt_boxes": ...,
    "gt_masks": ...,
    "gt_labels": ...,
}
```

建议先不要把视频时序样本混进来。
先把单帧 detector 调通，后面再决定是否把 LoRA 后 detector 接回 tracker。

## 9. 建议的工程实现方式

最清楚的做法是单独新增一个训练脚本，例如：

- `sam3_detr_exp/train_detr_lora.py`

当前已经进一步整理成：

- `sam3_detr_exp/train_detr_lora.py`
- `sam3_detr_exp/model/`
- `sam3_detr_exp/utils/`

建议职责拆分成四块：

### 1. 模型构建

基于 [modular_pipeline.py](../modular_pipeline.py) 的 `build_detector_model()`：

- 先加载模块化 detector
- 冻结默认不训练的参数
- 只对目标 transformer 层挂 LoRA

### 2. 提示编码

尽量复用 detector 当前的 prompt 流程：

- 文本提示走现有 text encoder
- 框提示走现有 geometry encoder

这样训练和推理路径一致，后面不会出现“训练用了一套，推理又是一套”。

### 3. 损失计算

对 detector 输出做：

- query matching
- 分类损失
- box 损失
- mask 损失

如果你暂时不想完整接原训练框架，也可以先做一个简化版：

- 只监督 top-k 预测
- 先验证 LoRA 是否能在小样本上过拟合

但正式训练前，还是建议回到稳定的 matching 逻辑。

### 4. 保存与加载

建议把保存物拆成两层：

1. 基础模块权重
   - 继续使用 `weights_modular/*.pt`
2. LoRA 增量权重
   - 单独保存为：
   - `weights_lora/detr_transformer_lora.pt`

这样你可以同时保留：

- 原始模块权重
- 不同任务的 LoRA 增量

加载顺序建议固定成：

1. load `weights_modular/*.pt`
2. attach LoRA modules
3. load LoRA adapter weights

## 10. 推理部署时怎么用 LoRA 结果

推理时有两种方式。

### 方式 A：保留 LoRA 结构，运行时加载适配器

优点：

- 最适合继续训练
- 不破坏原始模块权重
- 多个任务可以快速切换 adapter

缺点：

- 推理图里多一层 LoRA 逻辑

### 方式 B：把 LoRA 合并回基础线性层

优点：

- 推理更简单
- 更利于后续导出 ONNX / TensorRT

缺点：

- 不如 adapter 形式灵活
- 多任务切换不方便

如果你后面还要继续试多个数据集，建议先用方式 A。
如果最终要固化部署，再考虑 merge。

## 11. 和蒸馏的关系

LoRA 和蒸馏是能叠加的。

推荐顺序：

1. 先把 detector-only LoRA 跑通
2. 再考虑 teacher-student 蒸馏

蒸馏可加的位置：

- encoder memory
- decoder query features
- class logits
- box outputs
- mask logits

如果一开始 LoRA 都还没跑稳，不建议先上蒸馏。
不然问题会缠在一起，不容易定位。

## 12. 现在这套目录下的最小可行训练范围

如果按当前 `sam3_detr_exp` 目录状态，最小可行目标就是：

- 输入：
  - image
  - text prompt 或 box prompt
- 可训练：
  - `transformer_encoder`
  - `transformer_decoder`
- 可选联合训练：
  - `dot_product_scoring`
  - `segmentation_head`
- 输出：
  - boxes
  - scores
  - masks

这条线最短，也最符合你现在的模块化目的。

## 13. 不建议现在就做的事

下面这些事不是不能做，而是不适合作为第一步：

- 一上来对 `vision_backbone` 做 LoRA
- 一上来对 `text_encoder` 做 LoRA
- detector 和 tracker 一起联合训练
- 一边做 LoRA，一边做 ONNX 导出适配
- 一开始就把训练、蒸馏、视频传播三件事同时推进

先把 detector-only 跑通，你后面会轻松很多。

## 14. 推荐落地顺序

建议按这个顺序推进：

1. 先用 [train_detr_lora.py](../train_detr_lora.py) 跑通 Lightning `dry-run`
2. 基于 `build_detector_model()` 构建 detector-only 训练模型
3. 冻结除 `transformer.encoder/decoder` 之外的参数
4. 给目标线性层挂 LoRA
5. 在少量样本上做过拟合测试
6. 验证 `run_detr_prompt_inference.py` 推理路径可复用
7. 再决定是否解冻 `dot_product_scoring`
8. 再决定是否解冻 `segmentation_head`
9. 最后才考虑接回 video tracker

## 15. 一句话结论

如果你要做 SAM3 的 DETR LoRA，最好的切入点不是整模型，也不是 tracker，而是当前模块化 detector 里的：

- `transformer_encoder`
- `transformer_decoder`

先把这两个模块做成可插拔 LoRA，冻结 backbone 和 text encoder，围绕 detector-only 提示分割任务训练。

这条路工程成本最低，也最容易保持结构清楚、结果可控、后续还能继续模块化演化。

## 16. 当前目录里的训练入口

现在目录里已经补了一个最小可行训练入口：

- [train_detr_lora.py](../train_detr_lora.py)

这份脚本当前定位不是“正式大规模训练框架”，而是先把下面这条链打通：

- modular detector 加载
- LoRA 挂载
- detector-only 前向
- Hungarian matching
- 分类 / box / mask loss
- LoRA 增量权重保存

建议先这样验证完整的检测与分割训练链路：

```bash
./.venv/bin/python sam3_detr_exp/train_detr_lora.py \
  --data-yaml /path/to/data.yaml \
  --prompt-mode class_name \
  --train-dot-score \
  --train-seg-head \
  --max-train-samples 20 \
  --max-val-samples 10 \
  --dry-run
```

如果这个能正常跑完，说明这几个关键点已经对上了：

- 训练输入格式没错
- LoRA 已经挂到当前 transformer 实际参数上
- detector 输出和 matcher / loss 是对齐的
- 当前 modular pipeline 可以承接微调

这里的两个训练开关含义是：

- `--train-dot-score`：训练文本提示与目标查询之间的分类/匹配打分层，改善检测分类
- `--train-seg-head`：训练分割头，配合 mask loss 改善实例掩码

不加这两个开关时，分类打分层和分割头保持冻结；mask loss 仍可通过 LoRA 路径反向传播，但对当前同时要求检测和分割的任务，建议显式开启。

## 17. 通用训练方式

训练数据完全由 `--data-yaml` 指定，项目代码不绑定具体数据集、目录或类别。更换数据集时只需要更换 YAML。

对于从视频抽帧得到的数据，Train/Val 必须按视频或序列分组，不能把同一视频的相邻帧随机拆到两边，否则验证指标会因画面高度相似而虚高。具体划分脚本和统计应随数据集维护，不放进模型项目。

脚本当前处理逻辑是：

1. 读取同一张图下、同一类别的所有多边形实例
2. 把每个多边形 rasterize 成二值 mask
3. 从 mask 对应的 polygon 外接框生成 gt box
4. 用类别名作为文本提示，做 detector-only 训练

### 4×A800 80GB 推荐参数

`--batch-size` 是每个 GPU 的 batch size。使用 4 张卡且 `--batch-size 4` 时：

```text
global batch size = 4 GPUs × 4 samples/GPU = 16
```

建议从每卡 4 开始，而不是直接填满显存。分割样本的实例数和 mask 数量会变化，必须为高实例数 batch 和 CUDA 临时缓存保留余量。稳定运行数百 step 后，如果单卡峰值显存仍低于约 55GB，可以再尝试每卡 8。

推荐起步参数：

- `--devices 4`：使用 4 张 GPU
- `--batch-size 4`：每卡 4，全局 batch 16
- `--resolution 1008`：保持 SAM3 当前训练分辨率
- `--precision bf16-mixed`：A800 原生支持 BF16
- `--num-workers 8`：每个训练进程 8 个 worker，总计最多 32 个
- `--lr 2e-4`：LoRA 起始学习率
- `--weight-decay 1e-2`
- `--lora-rank 8`
- `--lora-alpha 16`
- `--lora-dropout 0.05`
- `--mask-weight 2.0`

4×A800 正式训练命令：

```bash
./.venv/bin/python sam3_detr_exp/train_detr_lora.py \
  --data-yaml /path/to/data.yaml \
  --prompt-mode class_name \
  --train-dot-score \
  --train-seg-head \
  --accelerator gpu \
  --devices 4 \
  --precision bf16-mixed \
  --resolution 1008 \
  --batch-size 4 \
  --num-workers 8 \
  --lr 2e-4 \
  --weight-decay 1e-2 \
  --mask-weight 2.0 \
  --lora-rank 8 \
  --lora-alpha 16 \
  --lora-dropout 0.05 \
  --epochs 50 \
  --save sam3_detr_exp/weights_lora/detr_lora.pt
```

`50` 个 epoch 是建议的首轮上限，不应只根据训练 loss 决定是否继续。应观察独立验证集的分类、box 和 mask loss；如果验证指标已经停止改善，就提前结束。若出现 OOM，优先把每卡 batch 从 `4` 降到 `2`，不要先降低输入分辨率。

每张图片会按其中出现的类别拆成文本提示样本，同一类别的多个实例一起参与 Hungarian matching、box loss 和 mask loss。因此当前训练同时覆盖检测框和实例分割，不是单纯的 box 监督。

## 18. 训练后怎么验证

训练完成后，直接用 detector-only 推理脚本加载 LoRA：

```bash
python sam3_detr_exp/run_detr_prompt_inference.py \
  --image assets/images/test_image.jpg \
  --text "class name from data yaml" \
  --lora sam3_detr_exp/weights_lora/detr_lora.pt \
  --output sam3_detr_exp/outputs/detr_lora.png
```

说明：

- `--lora` 指向 `train_detr_lora.py` 保存出来的增量权重
- 文本提示应优先使用 YAML `names` 中的训练类别名称
- `--train-dot-score` 和 `--train-seg-head` 训练出的额外权重也会包含在 LoRA checkpoint 中并由推理入口恢复

## 19. 当前代码结构

当前训练实现已经按职责拆开：

1. `train_detr_lora.py`
   - 只负责参数解析和 Lightning `Trainer.fit()`

2. `model/detr_lora_module.py`
   - `DetrLoraLightningModule`
   - 负责 training step / validation step / optimizer / LoRA checkpoint 保存

3. `utils/detr_lora_data.py`
   - YAML 驱动的 YOLO segmentation dataset
   - 从 YAML 的 `path/train/val/names` 解析数据和提示词
   - LightningDataModule

4. `utils/detr_lora_utils.py`
   - LoRA 挂载
   - detector 组装
   - prompt / target 构造
   - matcher + loss
   - LoRA save/load

当前训练框架版本：

- `lightning==2.6.5`
