# DETR LoRA 后续工作清单

当前多提示训练、数据集内负提示、通用域外负提示、SAM3 原生 loss、
auxiliary/O2M loss、固定验证 loss、best checkpoint 和 4 卡 DDP 训练均已实现。
本文档只保留尚未完成的工作。

## 1. 正式评估指标

训练选优目前仍以固定提示集合上的 `val/loss` 为准。需要增加与最终视觉效果更直接相关的指标：

- [ ] 每类 box AP、precision 和 recall。
- [ ] 每类 mask IoU 和 Dice。
- [ ] 数据集内空提示的 false-positive rate。
- [ ] 所有类别的 micro/macro 平均指标。
- [ ] 将上述指标按 epoch 写入训练日志。
- [ ] 评估使用综合检测/分割指标代替 `val/loss` 选择 best checkpoint。

评估必须固定以下条件，避免不同实验之间无法比较：

- 相同验证集及样本顺序。
- 相同类别提示词，不在验证阶段随机加入通用负提示。
- 相同置信度阈值、NMS 和后处理设置。
- 同时报告六项原始 loss、加权 loss 和最终任务指标。

## 2. 固定回归对比

- [ ] 建立统一脚本，在同一进程和相同参数下依次测试 base、旧 LoRA、新 LoRA。
- [ ] 固定使用 roadline 验证集和 `roadline/20260106` 前 10 张测试图。
- [ ] 输出逐图/逐类检测数量、分数、可视化及汇总 CSV。
- [ ] 加入检测框和 mask NMS，减少同一车道线的重复候选。
- [ ] 评估长实线实例是否需要 mask 合并或专门的后处理。

已有人工测试结果可作为参考，但尚未形成可重复执行的正式回归工具。

## 3. 优化器与训练参数消融

以下参数已有命令行选项，不需要修改代码：

- [ ] 使用 `--lora-rank 16 --lora-alpha 32 --lora-dropout 0.1` 训练对照实验。
- [ ] 使用 `--lr 1e-4` 训练对照实验。

以下能力尚未暴露或实现，需要先修改 `sam3_detr_exp` 内的训练代码：

- [ ] 增加 warmup steps 命令行参数，支持前 500 optimizer steps warmup。
- [ ] 增加 cosine learning-rate scheduler。
- [ ] 增加 `max_grad_norm` 命令行参数，并传给 Lightning Trainer 做梯度裁剪。
- [ ] 在日志中记录每个 optimizer step 的实际 learning rate。

建议按单变量消融实施，不要同时改变 LoRA 容量、学习率和调度策略：

1. 当前配置作为基线。
2. 仅改 rank/alpha/dropout。
3. 仅改初始 learning rate。
4. 仅加入 gradient clipping。
5. 再加入 warmup + cosine scheduler。

## 4. 自动化测试

### 数据集

- [ ] 一张多类别图片能生成正确数量的正提示和空 target 负提示。
- [ ] 无标注图片能够生成全部类别的空 target。
- [ ] 单类别图片和包含全部类别的图片均能正确加载。
- [ ] 通用负提示不会与数据集类别重复或共享冲突关键词。
- [ ] `max_train_samples` 和 `max_val_samples` 在 multi-prompt 模式下按图片计数。

### 目标与损失

- [ ] 整批 target 全为空时，IABCE/presence loss 有限且可反向传播。
- [ ] 空 target 的 box、GIoU、mask 和 Dice loss 为有限零值。
- [ ] 正负提示混合时，主输出、5 层 auxiliary 输出和 O2M 输出均能反向传播。
- [ ] 对 loss 子项的权重与总和关系增加断言测试。

### 验证确定性

- [ ] 同一 checkpoint 连续运行两次验证，逐项 loss 一致或误差在规定容差内。
- [ ] 4 卡验证结果与单卡验证结果在规定容差内一致。
- [ ] 验证阶段确认 `num_generic_negative_prompts=0`。

## 5. 推荐实施顺序

1. 建立统一 base/旧 LoRA/新 LoRA 回归脚本。
2. 增加 AP、recall、IoU、Dice 和空提示误检率。
3. 补齐 Dataset、空 target 和验证确定性测试。
4. 增加 gradient clipping、warmup 和 cosine scheduler。
5. 按单变量方式完成训练参数消融。
6. 根据任务指标决定是否改变 best checkpoint 的选取标准。
