# TinyViT Stage-3 道路标线轻量化蒸馏实验总览

本目录集中记录 TinyViT Stage-3 在道路标线场景上的轻量化实验。目标是在保留轻量模型结构和文本提示接口的前提下，尽可能接近 `Base + DETR LoRA` 的道路标线能力。

所有 TinyViT 相关代码、配置、日志、权重和测试都只放在本目录。测试产物位于 `tests/output/`，不提交 Git。

模型文件不能用“Stage-3基模大小 + LoRA checkpoint大小”直接相加：当前LoRA保存器重复保存了约
116 MiB的`weight.original`，而P5～P8只有约5.27 MiB是真正新增、无法折叠的结构。各阶段实际
文件组成、LoRA可合并范围、最终FP32/FP16推理体积和正式导出TODO见
[模型体积与合并分析](模型体积与合并分析.md)。

## 固定条件

- 学生基模：官方 Stage-3 TV-M，TinyViT-11M + MobileCLIP-S0。
- 教师：`sam3_detr_exp/weights_lora/roadline_r8_a16_lr2e4.best.pt`。
- 文本提示：道路标线配置中的 7 个固定类别，文本编码器冻结并实时提取特征。
- DETR：Encoder/Decoder 注意力和 FFN 使用 r8 LoRA。
- 头部：完整训练点积分类头和分割头。
- 基础监督：真实标签的 SAM3 O2O、辅助层和 O2M 损失。
- P0输出蒸馏：分类、presence、框和低分辨率 mask。
- 实际效果测试：相同的 10 张道路图片、相同文本提示、置信度阈值 0.5。

TinyViT 是一个图像编码器，内部按分辨率分为 stage 0～3。它包含卷积 stem、局部卷积、窗口注意力和 MLP，并非完全由自注意力组成。

## 目录与实验对应关系

```text
sam3_lightweight_tinyvit_stage3_distill_exp/
├── README.md                         # 本实验总览
├── 模型体积与合并分析.md             # 实际权重组成、合并方式和部署体积
├── train_p0_image_lora.py            # P0训练入口
├── image_lora.py                     # TinyViT图像LoRA注入
├── model_adapter.py                  # TinyViT Stage-3模型构建
├── configs/                          # P0配置
├── scripts/                          # P0脚本
├── p1_image_feature/                 # P1三尺度图像特征蒸馏
├── p2_image_stage123/                # P2图像LoRA扩展到stage 1/2/3
├── p3_all_r16/                       # P3图像与DETR LoRA统一提高到r16
├── p4_unfreeze_stage3_neck/          # P4严格对照：完整解冻Stage 3与neck
├── p5_dsconv_thin_line/              # P5方案：DSConv高分辨率细线分支
├── p6_multiscale_dsconv/             # P6：冻结P5并新增Stage 1 DSConv分支
├── p7_highres_fpn/                   # P7：Stage1/2方向特征直连高/中分辨率FPN
├── p8_input_line_branch/             # P8：输入侧504分辨率细线提取
├── tests/                            # 统一评测代码
│   └── output/                       # 测试结果，不提交Git
├── cache/                            # 教师缓存，不提交Git
├── logs/                             # 训练日志，不提交Git
└── weights/                          # LoRA与训练头权重，不提交Git
```

外部对照实验：

| 对照 | 目录 | 用途 |
|---|---|---|
| Base + DETR LoRA | `../sam3_detr_exp/` | 当前道路标线效果上限和教师模型 |
| EfficientViT Stage-3 LoRA | `../sam3_lightweight_stage3_exp/` | 早期轻量骨干基线 |
| P0教师输出缓存 | `../sam3_lightweight_stage3_distill_exp/cache/p0_teacher/` | Base最终输出软目标，供P0/P1/P2复用 |

## 实验路线

### P0：TinyViT Stage 2/3 r8 LoRA + 最终输出蒸馏

对应位置：

- 训练代码：`train_p0_image_lora.py`
- 配置：`configs/p0_image_lora_r8.yaml`
- 脚本：`scripts/train_p0_image_lora.sh`
- 最佳权重：`weights/p0_image_lora_r8.best.pt`
- 日志：`logs/p0/lightning_logs/version_3/`
- 测试：`tests/output/tinyvit_vs_base_detr_first10_threshold_05/`

训练内容：

- TinyViT stage 2、3 的 attention `qkv/proj` 和 MLP `fc1/fc2` 使用 r8 LoRA。
- DETR Encoder/Decoder使用 r8 LoRA。
- 训练点积分类头和分割头。
- 使用真实标签监督和 Base最终输出KD。

最佳训练点为 epoch 6：

| 指标 | 数值 |
|---|---:|
| `val/supervised` | 6.2367 |
| `val/kd` | 3.7750 |
| `val/loss` | 10.0117 |

测试项目与结果：

| 测试项目 | 测试配置 | 结果 |
|---|---|---|
| 训练收敛测试 | 真实标签监督 + Base最终输出KD | epoch 6取得最佳`val/loss=10.0117` |
| 白实线分割 | 固定10张图，阈值0.5 | IoU 0.5187，Recall 0.6676 |
| 白虚线分割 | 固定10张图，阈值0.5 | IoU 0.3400，Recall 0.4139 |
| 负类误检观察 | 同一测试集中的护栏、斑马线等无真值类别 | 护栏误检10张，斑马线误检10张 |
| Base差距 | 与`../sam3_detr_exp/`同配置比较 | 两类平均IoU 0.4293，Base为0.7021 |

结论：P0训练稳定，TinyViT已经具备可用的道路标线能力，但白虚线召回明显不足，与Base差距较大。

### P1：增加三尺度图像特征蒸馏

对应目录：`p1_image_feature/`

- 说明：`p1_image_feature/README.md`
- 教师缓存：`cache/p1_image_features/`
- 训练脚本：`p1_image_feature/scripts/train_p1_image_feature.sh`
- 最佳权重：`weights/p1_image_feature_r8.best.pt`
- 日志：`logs/p1_image_feature/lightning_logs/version_1/`
- 测试：`tests/output/p1_image_feature_vs_base_first10_threshold_05/`

在 P0基础上增加 Base与TinyViT三层同形FPN特征对齐：

```text
256×288×288
256×144×144
256×72×72
```

教师缓存统一平均池化4倍，并使用逐像素归一化 cosine loss；真实标注区域权重为背景的4倍。TinyViT仍只在stage 2、3训练r8 LoRA。

最佳训练点为 epoch 6：

| 指标 | P0最佳 | P1最佳 | 变化 |
|---|---:|---:|---:|
| `val/supervised` | 6.2367 | 6.1320 | -0.1047 |
| 原P0输出KD | 3.7750 | 约3.7527 | -0.0223 |
| 图像特征KD | 无 | 0.3424 | 新增项 |
| 扣除图像KD后的可比损失 | 10.0117 | 约9.8848 | -0.1269 |

测试项目与结果：

| 测试项目 | 测试配置 | 结果 |
|---|---|---|
| 图像特征KD收敛 | 三尺度FPN、4倍池化、前景4倍权重 | 最佳图像特征KD 0.3424 |
| 白实线分割 | 固定10张图，阈值0.5 | IoU 0.5343，Recall 0.6863 |
| 白虚线分割 | 固定10张图，阈值0.5 | IoU 0.3490，Recall 0.4279 |
| P0增益 | 与P0同配置比较 | 白实线IoU +0.0156，白虚线IoU +0.0090 |
| 负类误检观察 | 同一测试集中的护栏、斑马线等无真值类别 | 护栏误检由10张降至5张；斑马线仍误检10张 |

结论：三尺度图像特征蒸馏有效，但实际IoU提升有限，没有接近Base。r8图像LoRA可以向教师特征靠近，但容量或覆盖范围仍受限。

### P2：图像LoRA扩展到Stage 1/2/3

对应目录：`p2_image_stage123/`

- 说明：`p2_image_stage123/README.md`
- 训练脚本：`p2_image_stage123/scripts/train_p2_image_stage123.sh`
- 当前最佳权重：`weights/p2_image_stage123_r8.best.pt`
- 日志：`logs/p2_image_stage123/lightning_logs/version_0/`

P2从P1最佳权重继续训练：

- stage 2、3 LoRA、DETR LoRA和训练头继承P1。
- stage 1新增8个r8 LoRA模块，初始增量为零。
- 新增可训练参数只有32,768个。
- 蒸馏损失、数据和学习率保持不变。

截至2026-08-20共完成6个epoch后停止，最佳点为epoch 5：

| 指标 | P1最佳 | P2 epoch 0 | P2最佳epoch 5 |
|---|---:|---:|---:|
| `val/supervised` | 6.1320 | 6.1040 | 6.0514 |
| 图像特征KD | 0.3424 | 0.3420 | 0.3320 |
| `val/loss` | 10.2271 | 10.1924 | 10.1169 |
| 扣除图像KD后的可比损失 | 9.8848 | 9.8504 | 9.7849 |

阶段结论：训练与实际测试均有稳定提升，但没有质的飞跃。P2是在P1最佳上继续训练，因此下降不能严格全部归因于stage 1；若要证明stage 1的独立贡献，需要增加一个从同一P1最佳权重继续训练、但仍只覆盖stage 2/3的对照组。统一测试中，P2相对P1的白实线IoU提高0.0111，白虚线IoU提高0.0162，白虚线Recall提高0.0182。

测试项目与结果：

| 测试项目 | 测试配置 | 结果 |
|---|---|---|
| Stage 1扩展训练 | 从P1最佳继续训练，共6个epoch | 最佳epoch 5，`val/supervised=6.0514` |
| 白实线分割 | 固定10张图，阈值0.5 | IoU 0.5454，Precision 0.6974，Recall 0.7145 |
| 白虚线分割 | 固定10张图，阈值0.5 | IoU 0.3652，Precision 0.6683，Recall 0.4461 |
| P1增益 | 与P1同配置比较 | 白实线IoU +0.0111，白虚线IoU +0.0162，白虚线Recall +0.0182 |
| 负类误检观察 | 同一测试集中的无真值类别 | 斑马线误检10张，护栏误检0张 |

测试结论：P2在当时是效果最好的TinyViT方案，但相对P1只有小幅稳定提升；由于它从P1最佳点续训，结果同时包含新增Stage 1 LoRA与额外训练轮数的影响。P3完成后平均IoU略高于P2，但差值仅0.0010。

### P3：图像与DETR LoRA统一提高到r16

对应目录：`p3_all_r16/`

- 转换脚本：`p3_all_r16/scripts/convert_all_lora_rank.sh`
- 训练脚本：`p3_all_r16/scripts/train_p3_all_r16.sh`
- 初始化权重：`weights/p3_all_r16_init.pt`
- 正式输出：`weights/p3_all_r16.best.pt`
- 日志：`logs/p3_all_r16/`

P3先把P2最佳权重中的图像r8和DETR r8 LoRA增量分别合并进对应基础权重，再在TinyViT stage 1/2/3与DETR Encoder/Decoder统一挂载新初始化的r16、alpha32 LoRA。点积分类头、分割头和蒸馏配置保持不变。全r16方案已完成转换验证、10轮正式训练和统一10图评测。

当前已完成的测试项目与结果：

| 测试项目 | 测试配置 | 结果 |
|---|---|---|
| 权重转换完整性 | 合并图像r8与DETR r8，再统一挂载r16 | 合并/挂载图像40个、DETR 84个，生成`weights/p3_all_r16_init.pt` |
| 转换前后一致性 | 同一张图片、阈值0.5，比较P2最佳与P3初始化 | 白实线IoU 0.5373→0.5334；白虚线0.3978→0.3989，没有能力断裂 |
| 单步训练冒烟测试 | 1个训练step + 1个验证step | 训练与验证通过，`val/loss=11.1124` |
| 梯度检查 | 图像r16、DETR r16和两个输出头 | 梯度范数分别为56.81、63.03、76.99 |
| 参数量检查 | 图像与DETR均为r16 | 总参数105,844,922；可训练参数5,792,001 |
| 正式训练 | 从P2最佳转换权重继续，共10轮 | 最佳epoch 3，`val/loss=10.0335`，`val/supervised=5.9835` |
| 白实线分割 | 固定10张图，阈值0.5 | IoU 0.5487，Precision 0.6950，Recall 0.7228 |
| 白虚线分割 | 固定10张图，阈值0.5 | IoU 0.3639，Precision 0.6659，Recall 0.4452 |
| 负类误检观察 | 同一测试集中的无真值类别 | 斑马线误检10张，护栏误检1张 |

阶段结论：P3相对P2的`val/loss`下降0.0834、`val/supervised`下降0.0679，但平均IoU只提高0.0010。白实线IoU提高0.0033、Recall提高0.0084；白虚线IoU下降0.0013、Recall下降0.0008。提高LoRA秩改善了训练目标，却没有解决白虚线召回，也没有产生有意义的实际效果提升。

## 统一实际效果对比

以下结果来自同一批10张图片、阈值0.5。该批图片只有白实线和白虚线真值，因此其他类别只能用于观察误检，不能用于评估正样本IoU。

| 模型 | 对应目录 | 白实线IoU | 白实线Recall | 白虚线IoU | 白虚线Recall | 两类平均IoU |
|---|---|---:|---:|---:|---:|---:|
| EfficientViT Stage-3 LoRA | `../sam3_lightweight_stage3_exp/` | 0.3905 | 0.5137 | 0.2153 | 0.2403 | 0.3029 |
| TinyViT P0 | 本目录根部 | 0.5187 | 0.6676 | 0.3400 | 0.4139 | 0.4293 |
| TinyViT P1图像特征KD | `p1_image_feature/` | 0.5343 | 0.6863 | 0.3490 | 0.4279 | 0.4417 |
| TinyViT P2 Stage 1/2/3 | `p2_image_stage123/` | 0.5454 | 0.7145 | 0.3652 | 0.4461 | 0.4553 |
| TinyViT P3全LoRA r16 | `p3_all_r16/` | 0.5487 | 0.7228 | 0.3639 | 0.4452 | 0.4563 |
| P4 Control最佳（epoch 0） | `p4_unfreeze_stage3_neck/` | 0.5411 | 0.7200 | 0.3680 | 0.4550 | 0.4546 |
| P4解冻最佳（epoch 1） | `p4_unfreeze_stage3_neck/` | 0.5180 | 0.6801 | 0.3664 | 0.4652 | 0.4422 |
| Base + DETR LoRA | `../sam3_detr_exp/` | 0.7235 | 0.7883 | 0.6808 | 0.8226 | 0.7021 |

P0到P1的实际变化：

- 白实线IoU相对提升约3.0%。
- 白虚线IoU相对提升约2.6%。
- 白虚线Recall相对提升约3.4%。
- 护栏负类误检数从10降到5；斑马线误检仍为10。

总体结论：TinyViT明显优于此前EfficientViT轻量基线，图像特征蒸馏和扩大图像LoRA覆盖范围带来稳定正收益；但P3把全部LoRA提高到r16后几乎没有继续提升，P4完整解冻Stage 3与neck后平均IoU反而下降，说明限制效果的主要因素既不是低秩容量，也不能通过直接放开高层视觉参数解决。轻量模型与Base仍有明显差距，尤其是白虚线Recall。

## Loss比较注意事项

- Base的`val/loss`只包含SAM3监督损失。
- P0的`val/loss = supervised + 最终输出KD`。
- P1/P2/P3的`val/loss = supervised + 最终输出KD + 图像特征KD`。
- 不同实验的验证batch size曾经不同，原始损失还存在批量聚合尺度差异。
- 因此跨实验优先比较`val/supervised`、扣除新增KD后的趋势和统一实际IoU，不直接比较`val/loss`绝对值。

## 当前阶段结论与下一步

P2之后的结构实验目前把两类平均IoU从0.4553提高到P8第1轮的0.6155：

| 实验 | 训练起点 | 正式训练 | 实线IoU | 虚线IoU | 平均IoU | 结论 |
|---|---|---:|---:|---:|---:|---|
| P2 | P1最佳 | 10轮 | 0.5454 | 0.3652 | 0.4553 | Stage 1/2/3图像LoRA基线 |
| P5 | P2最佳 | 20轮 | 0.5746 | 0.4967 | 0.5357 | Stage 2细线旁路明显有效 |
| P6 | P5 epoch 15 | 10轮 | 0.6214 | 0.5610 | 0.5912 | Stage 1+2双尺度继续显著提升 |
| P7 | P6 epoch 8 | 10轮 | 0.6319 | 0.5687 | 0.6003 | 中分辨率直连小幅有效，高分辨率简单投影基本未采用 |
| P8（阶段结果） | P7 epoch 9 | 计划5轮，已完成1轮 | 0.6511 | 0.5800 | 0.6155 | 输入侧504分辨率分支首轮已超过0.610验收线 |
| Base + DETR LoRA | Base权重 | 10轮内提前停止 | 0.7235 | 0.6808 | 0.7021 | 当前教师与效果上限 |

P7相对P6平均IoU只提高0.0091，低于预设的0.02明确增益门槛；最终`gate_high=-0.0171`、`gate_mid=0.7620`，说明Stage 1方向特征简单插值到`288×288`没有被模型有效采用，Stage 2到`144×144`才是主要增益来源。

P8输入侧细线分支已经实现并开始训练：使用`PixelUnshuffle(2)`把RGB无损重排到504分辨率，在大幅下采样前运行低通道DSConv，再通过Stage 2语义门控融合到高分辨率FPN。第1轮平均IoU为0.6155，已超过0.610验收线；仍需完成计划的5轮后选择正式最佳权重。代码已预留参数匹配的`1×9 + 9×1`普通长条卷积实现，因为P5-B尚未执行，目前结果不能证明动态蛇形采样本身优于普通长条卷积。

### P4：解冻Stage 3与neck严格对照

对应目录：`p4_unfreeze_stage3_neck/`

P4已经完成实现、两组完整3轮训练及逐轮统一评测。Control与Unfreeze都从P2最佳权重开始并保持数据与蒸馏损失一致：

- Control保持P2结构，继续训练Stage 1/2/3图像r8 LoRA、DETR r8 LoRA和两个输出头。
- Unfreeze合并并移除Stage 3的8个图像LoRA，完整解冻Stage 3与FPN neck；Stage 1/2图像r8 LoRA、DETR r8 LoRA和输出头保持不变。
- Unfreeze可训练参数为17,164,125，其中Stage 3为4,839,772，neck为7,802,112。
- 专用保存器会保存完整Stage 3与neck权重，不会只保存LoRA而丢失解冻参数。

判定门槛：Unfreeze相对Control平均IoU至少提高0.01，或白虚线Recall至少提高0.02，且负类误检不能明显增加。

实际结果：Control最佳为epoch 0，平均IoU 0.4546；Unfreeze最佳为epoch 1，平均IoU 0.4422，epoch 2又降至0.4395。Unfreeze相对Control平均IoU下降0.0124，白虚线Recall只提高0.0102，因此未达到门槛。虽然Unfreeze最低验证loss从Control的10.1254降至9.9673，但实际分割指标下降，最终判定P4无效。

### P5：Stage 2 DSConv高分辨率细线分支

对应目录：`p5_dsconv_thin_line/`

P5-A已经完成20轮正式训练和统一测试。它从P2最佳权重出发，在TinyViT Stage 2高分辨率特征与FPN之间增加零门控细线旁路。P2已有LoRA增量保留在前向中，但全部冻结；仅新增896,019个可训练参数。

P5-A冻结TinyViT、全部已有LoRA、neck、DETR和输出头，只训练横向/纵向DSConv、偏移预测器、融合投影和零门控。原计划在P5-A有效后执行普通卷积P5-B消融，但为了优先追求最终效果，当时直接进入P6；因此普通卷积严格对照仍是未完成项。

P5 epoch 15为验证损失与实际任务共同最佳点：白实线IoU 0.5746、白虚线IoU 0.4967、两类平均IoU 0.5357。相对P2平均IoU提高0.0804，主要增益来自白虚线，证明高分辨率细线分支具有明确价值；但仍低于Base的0.7021。完整设计、训练记录和风险见P5目录README。

### P6：Stage 1 + Stage 2双尺度DSConv

对应目录：`p6_multiscale_dsconv/`

P6直接以`weights/p5a_dsconv_frozen_20ep.epoch15.pt`为起点，跳过只服务于机制归因的普通卷积消融。已训练的Stage 2分支及全部P5参数冻结，新增`128×126×126` Stage 1特征分支；分支中间通道为64，新增且可训练参数301,587。

实现和冒烟验证包括：

- Stage 1零门控初始化与P5 epoch 15输出逐元素完全一致，最大绝对误差为0。
- P5 Stage 2门控从checkpoint精确恢复为-1.24412823。
- 第一个训练step只有Stage 1门控获得梯度；第二个step后偏移、投影和融合参数均获得非零梯度。
- 统一评测脚本可以自动挂载两个DSConv分支并完成推理。

P6完成10轮正式训练和逐轮统一评测。实际IoU最佳为epoch 8：实线0.6214、虚线0.5610、平均0.5912，相对P5提高0.0555，超过预设验收线；最低验证loss在epoch 9，但实际平均IoU略低，因此P7固定使用epoch 8初始化。

### P7：高分辨率FPN直接融合

对应目录：`p7_highres_fpn/`

P7从P6实际IoU最佳的`weights/p6_stage1_frozen_p5.epoch8.pt`开始，复用冻结的横/纵DSConv方向特征，分别新增零门控投影到`256×288×288`和`256×144×144`的FPN层。原P6低分辨率残差保留，因此P7初始化不损失P6已有能力。第一阶段只训练约10万参数的高、中分辨率适配器，详细结构与训练命令见本实验目录README。

P7完成10轮正式训练，epoch 9实际平均IoU最佳：实线0.6319、虚线0.5687、平均0.6003，相对P6提高0.0091。训练后高分辨率门控仅为-0.0171，中分辨率门控为0.7620；结论是中分辨率直连有效，但Stage 1特征经过简单`1×1`投影和插值后没有被充分利用，P7未达到原定0.6112验收线，不继续追加轮数。

### P8：输入侧504分辨率细线分支

对应目录：`p8_input_line_branch/`

P8从P7 epoch 9开始，使用PixelUnshuffle把1008输入无损重排到504分辨率，在进一步下采样前用低通道横/纵DSConv提取细线，再通过Stage 2语义门控融合到`288×288`FPN。第一阶段冻结P7全部已有参数，只训练新增分支5轮。当前实际运行使用每卡batch 1，已经完成epoch 0：`val/loss=9.1876`、`val/supervised=5.3093`；统一10图白实线IoU/Recall为0.6511/0.7758，白虚线为0.5800/0.6737，平均IoU为0.6155，相对P7提高0.0152。该结果已超过0.610门槛，但训练尚未完成，不能当作最终最佳值。训练入口同时预留普通`1×9 + 9×1`长条卷积对照，详细结构和命令见P8目录README。

## 常用命令

P0四卡训练：

```bash
bash sam3_lightweight_tinyvit_stage3_distill_exp/scripts/train_p0_image_lora.sh
```

P1教师图像特征四卡缓存：

```bash
bash sam3_lightweight_tinyvit_stage3_distill_exp/p1_image_feature/scripts/cache_teacher_features_4gpu.sh
```

P1四卡训练：

```bash
bash sam3_lightweight_tinyvit_stage3_distill_exp/p1_image_feature/scripts/train_p1_image_feature.sh
```

P2四卡训练：

```bash
bash sam3_lightweight_tinyvit_stage3_distill_exp/p2_image_stage123/scripts/train_p2_image_stage123.sh
```

P3全LoRA r16权重转换与四卡训练：

```bash
bash sam3_lightweight_tinyvit_stage3_distill_exp/p3_all_r16/scripts/convert_all_lora_rank.sh
bash sam3_lightweight_tinyvit_stage3_distill_exp/p3_all_r16/scripts/train_p3_all_r16.sh
```
