# DETR LoRA 后续工作清单

当前多提示训练、数据集内负提示、通用域外负提示、SAM3 原生 loss、
auxiliary/O2M loss、固定验证 loss、best checkpoint 和 4 卡 DDP 训练均已实现。
本文档只保留尚未完成的工作。

## 0. 负提示词来源、退化证据与待验证假设

### 0.1 来源辨析

需要区分“原始 SAM3 的负查询机制”和本项目采用过的“硬编码副提示词采样策略”：

- 原始 SAM3 本身支持负文本查询和 Presence 判断，训练配置中也存在
  `include_negatives: true`。这些负查询来自数据集及查询采样流程，用于学习“当前文本目标
  是否存在”。
- “每张图片补齐数据集内缺失类别，并额外随机加入 2 个
  `person`、`dog`、`cat`、`chair` 等通用域外词作为负提示”这一具体实现，来源于外部
  `SAM3_LoRA` 项目的自定义脚本
  `/slow_disk/ccl/codes/SAM3_LoRA/train_sam3_lora_with_categories.py`，不是原始 SAM3
  规定的固定训练方式。
- 原始 SAM3 的大规模多类别训练中，同一通用概念会在一些图片中作为负查询、在另一些图片中
  获得正样本监督。道路标线小数据集中的硬编码通用负提示没有对应正样本，监督明显不对称。

因此，不能把当前开放类别能力退化简单归因于“SAM3 使用了负查询”或 Presence loss；目前首先
需要验证的是小数据集里高比例、只有负样本的通用副提示是否造成了灾难性类别抑制。

### 0.2 已有对照证据

固定道路图片、文本提示 `car`、置信度阈值 0.5 的历史结果如下：

| Base 权重 | 提示构造 | `car` 检测数 | 结论 |
|---|---|---:|---|
| 早期 `detr_lora.pt` | 单正提示，早期简化 loss | 11 | 开放类别能力仍保留 |
| 早期 `roadline_sam3_loss_lora.best.pt` | 单正提示，SAM3 原生 loss/Presence | 20 | 原生 loss/Presence 本身没有导致能力消失 |
| 正式 `roadline_r8_a16_lr2e4.best.pt` | 多提示，包含数据集内与通用域外负提示 | 0 | 开放类别能力已经丢失 |
| 回溯消融`roadline_r8_a16_lr2e4_no_generic_negatives.best.pt` | 多提示，只保留数据集内负提示 | 20 | 关闭域外纯负提示后恢复`car` |

正式训练的一个已记录批次由 2 张图片展开为 18 个文本查询，其中 4 个正提示、10 个数据集内
负提示、4 个通用域外负提示，负查询占 77.8%。这些结果形成了较强相关证据，但尚不能替代严格
单变量实验；分类头持续更新、训练轮数和多提示批次展开也可能共同作用。

### 0.3 未来必须完成的单变量实验

第一个正式消融已经在[域外负提示消融目录](../negative_prompt_ablation/README.md)完成：保持正式
`roadline_r8_a16_lr2e4`的多提示训练条件不变，仅通过`--num-generic-negatives 0`关闭域外通用
纯负提示。20轮最佳`val/loss=4.1741`；统一10图平均IoU为0.7483，高于旧教师的0.7021；同图
`car`检测由0恢复到20。域外纯负提示不是道路标线训练所必需，并且是当前已验证的跨类别退化主因。

所有实验必须从同一个未专项化 Base checkpoint 开始，固定数据划分、随机种子、训练轮数、
学习率、LoRA 配置和 batch 中的图片数，只改变提示构造：

- [ ] A：单正提示基线，不加入任何负提示。
- [ ] B：单正提示 + SAM3 原生 loss/Presence，复现早期能力保持结果。
- [x] C：正提示 + 数据集内缺失道路标线类别，禁止通用域外负提示。已完成，平均IoU 0.7483，`car=20`。
- [x] D：在 C 的基础上加入通用域外负提示，复现正式训练策略。历史完整20轮基线最低loss 4.2201，旧正式教师`car=0`。
- [ ] E：通用类别同时加入少量正样本或旧 Base 教师的保持蒸馏，验证能否避免类别抑制。
- [ ] F：在 C/D 中分别冻结与训练点积分类头，分离分类头漂移的影响。
- [ ] 扫描每图负提示数量及正负比例，至少比较 0、1、2 个通用负提示和受控的 1:1、1:2
  正负比例。

每个 checkpoint 除道路标线 IoU/Recall 外，必须固定测试 `car`、`person`、`dog` 等训练中出现过
的通用负词，以及若干从未作为副提示出现的开放类别，并记录阈值 0.1/0.3/0.5 下的检测数、
最高置信度和 Presence logits。验收目标是：道路标线指标不明显下降，同时开放类别相对初始 Base
不发生系统性归零。完成上述实验前，正式训练默认禁用纯负的通用域外副提示。

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
