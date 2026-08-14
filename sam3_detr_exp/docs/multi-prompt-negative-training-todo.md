# Multi-prompt 与负提示训练 TODO

## 目标

让 `train_detr_lora.py` 通过数据 YAML 配置单图多提示训练，行为与本地
`/slow_disk/ccl/codes/SAM3_LoRA/` 的 roadline 训练方式保持一致：

- 每张图片只计算一次视觉 backbone。
- YAML 中的所有数据集类别都作为文本提示。
- 图片中存在的类别使用对应实例作为正 target。
- 图片中不存在的类别使用空 target，作为数据集内负提示。
- 每张图片额外随机采样若干通用域外负提示，target 为空。
- 训练策略由数据 YAML 决定，换数据集时不再修改训练代码。

本 TODO 只记录未来改造方案，当前尚未实施。

## 建议的 YAML 格式

```yaml
path: /slow_disk/ccl/data/roadline20251023
train: video_disjoint/train
val: video_disjoint/val

names:
  0: white solid lane line
  1: yellow solid lane line
  2: white dashed lane line
  3: yellow dashed lane line
  4: zebra crossing
  5: lane barrier
  6: road teeth marking

prompt_training:
  mode: multi_prompt
  num_negatives: 2
  generic_negatives:
    - person
    - dog
    - cat
    - chair
    - table
    - bottle
    - phone
    - bird
    - flower
    - window
    - door
```

未设置 `prompt_training` 或设置 `mode: single_prompt` 时，应保持现有行为，
确保旧数据 YAML 和旧命令兼容。

## 每张图片的提示组成

以 7 类 roadline 数据为例：

1. 固定生成全部 7 个数据集内类别提示。
2. 图中存在的类别关联该类别的 box 与 mask。
3. 图中不存在的类别关联零个 box/mask，并设置为 exhaustive 空 target。
4. 从 `generic_negatives` 随机采样 `num_negatives: 2` 个提示，关联空 target。

因此每张图片通常产生 9 个提示：7 个数据集类别提示和 2 个通用负提示。

通用负提示必须确保在图像中确实不存在。roadline 数据不宜使用 `car`、
`vehicle`、`road`、`asphalt` 等高概率真实出现的概念，否则会形成错误监督。

## 数据结构改造

- [ ] 扩展 `YoloDatasetConfig`，解析并保存 `prompt_training` 配置。
- [ ] 将 `YoloSegmentationDataset.records` 从“图片-类别”记录改为“一张图片一条记录”。
- [ ] 一次解析并保存图片内所有类别的 polygon、box 和 mask。
- [ ] 扩展或替换 `Sample`，使其保存唯一图片及多个 `PromptTarget`。
- [ ] `PromptTarget` 至少包含 `text_prompt`、`gt_boxes`、`gt_masks` 和正/负类型。
- [ ] 允许 `gt_boxes` 为 `[0, 4]`、`gt_masks` 为 `[0, H, W]` 的空 target。
- [ ] `max_train_samples` / `max_val_samples` 明确定义为图片数量，不再是展开后的 prompt 数量。

## Batch 与模型输入改造

- [ ] collate 后保留 `B` 张唯一图片，同时将其提示展开成 `Q` 个 query。
- [ ] `images` 的形状保持 `[B, 3, H, W]`，文本列表长度为 `Q`。
- [ ] 构造真实映射，例如两张图共五个提示时：

  ```text
  img_ids  = [0, 0, 0, 1, 1]
  text_ids = [0, 1, 2, 3, 4]
  ```

- [ ] 修改 `make_find_stage`，接收上述 `img_ids` / `text_ids`，而不是默认一一对应。
- [ ] 图像 backbone 只处理 `B` 张图片；文本 backbone 处理 `Q` 个提示。
- [ ] 检查 segmentation head 是否通过 `img_ids` 正确复用对应图片特征。
- [ ] batch size 的含义统一为“每卡图片数”，启动时打印唯一图片数、prompt 数和全局有效 prompt batch。

## Target 与 loss 改造

- [ ] `build_targets` 按展开后的 `Q` 个提示构造 target。
- [ ] `num_boxes` 支持零目标提示。
- [ ] padded box/object ID 的维度在整批全为空时也必须合法，避免 `max()` 或拼接失败。
- [ ] mask 打包支持正负提示混合，并保持 matcher 返回索引与 packed mask 一致。
- [ ] `is_exhaustive=True`，使缺失类别和通用负提示明确表示“该提示没有目标”。
- [ ] 验证官方 IABCE/presence loss 对空 target 的分类监督有效。
- [ ] 验证 box、GIoU、mask、Dice loss 对空 target 返回有限的零值。
- [ ] 验证主输出、5 层 auxiliary 输出和 O2M 输出都支持正负提示混合。

## 通用负提示规则

- [ ] 过滤与任一数据集类别共享关键词的通用负提示，避免文本概念冲突。
- [ ] 去除与当前图片正类提示完全相同或近似重复的负提示。
- [ ] 训练阶段使用受 seed 控制的随机采样，DDP 下各 rank 行为可复现。
- [ ] 验证阶段不得随机变化：使用固定采样或只验证全部数据集内类别。
- [ ] 日志分别记录正提示、数据集内负提示、通用负提示的数量。

## 验证策略

建议验证阶段对每张图固定使用 YAML 的全部数据集类别，不加入随机通用负提示。
这样各 epoch 的 `val/loss` 口径稳定，并同时考察：

- 正类检出和分割质量。
- 缺失类别误检抑制。
- 白/黄、实线/虚线等相近类别的区分。
- presence 与分类分数的校准。

除 `val/loss` 外，后续应补充更直观的指标：

- [ ] 每类 box AP / recall。
- [ ] 每类 mask IoU / Dice。
- [ ] 空提示 false-positive rate。
- [ ] micro/macro 平均指标。
- [ ] 相同图片上的 base、旧 LoRA、新 LoRA 可视化对比。

最佳模型仍按固定口径的 `val/loss` 保存；具备 AP/IoU 后，应评估是否改用综合指标选 best。

## 显存与推荐起始参数

多提示会复用视觉 backbone，但 DETR、matcher 和 mask decoder 仍按提示数量扩展。
7 个类别加 2 个通用负提示时，每张图片约对应 9 个 prompt，因此不能沿用当前每卡
图片 batch 8。

建议首次 smoke test：

```text
GPU：4 × A800 80G
每卡图片 batch：1
全局图片 batch：4
每卡 prompt 数：约 9
```

确认峰值显存后再尝试每卡图片 batch 2。训练日志应输出每批实际 prompt 数，因为不同
数据集类别数量或配置会改变显存占用。

## 优化器对齐（可单独实施）

开源 roadline 配置还使用了以下训练设置，它们不属于多提示功能本身，但做严格对照实验时
应作为独立变量加入：

- [ ] LoRA rank 16、alpha 32、dropout 0.1。
- [ ] learning rate `1e-4`。
- [ ] warmup 500 optimizer steps。
- [ ] cosine learning-rate scheduler。
- [ ] gradient clipping `max_grad_norm: 1.0`。

不要同时改变数据组织、LoRA 容量和优化器后直接归因。建议先仅加入多提示和负提示，
其他参数保持不变做消融对比。

## 测试与验收

- [ ] Dataset 单元测试：一张多类别图片生成正确的正/负提示及目标。
- [ ] Dataset 单元测试：无标注图片、单类别图片、全类别图片均可加载。
- [ ] Target 单元测试：整批全为空时 loss 有限且可反向传播。
- [ ] 单卡 1 batch forward/backward smoke test。
- [ ] 单卡正负提示混合的 auxiliary/O2M 分支 smoke test。
- [ ] 4 卡 DDP 10 batch smoke test，无 unused parameter、NaN 或 OOM。
- [ ] 验证集重复运行两次，`val/loss` 在相同 checkpoint 下完全一致或仅有可解释的数值误差。
- [ ] 日志显示唯一图片数、总 prompt 数、正提示数、两类负提示数。
- [ ] 对 roadline 验证集和 `roadline_20260106` 固定图片做 base/旧 LoRA/新 LoRA 对比。

## 推荐实施顺序

1. YAML 解析及旧格式兼容。
2. 一图一记录的数据集结构。
3. 单图多正提示与 `img_ids` 映射。
4. 数据集内缺失类别负提示。
5. 空 target 的 matcher/loss 测试。
6. 固定口径的多类别验证。
7. 通用负提示及冲突过滤。
8. 单卡、4 卡 smoke test。
9. 与当前最佳 LoRA 做消融训练和可视化评估。
