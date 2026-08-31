# TinyViT Stage-3 道路标线轻量化蒸馏实验总览

本目录集中记录 TinyViT Stage-3 在道路标线场景上的轻量化实验。目标是在保留轻量模型结构和文本提示接口的前提下，尽可能接近 `Base + DETR LoRA` 的道路标线能力。

所有 TinyViT 相关代码、配置、日志、权重和测试都只放在本目录。测试产物位于 `tests/output/`，不提交 Git。

模型文件不能用“Stage-3基模大小 + LoRA checkpoint大小”直接相加：当前LoRA保存器重复保存了约
116 MiB的`weight.original`，而P5～P8只有约5.27 MiB是真正新增、无法折叠的结构。各阶段实际
文件组成、LoRA可合并范围、最终FP32/FP16推理体积和正式导出TODO见
[模型体积与合并分析](模型体积与合并分析.md)。

## 固定条件

- 学生基模：官方 Stage-3 TV-M，TinyViT-11M + MobileCLIP-S0。
- 教师：P0～P8使用历史`roadline_r8_a16_lr2e4.best.pt`；P9改用“Base回溯消融：无域外负提示”
  训练得到的`roadline_r8_a16_lr2e4_no_generic_negatives.best.pt`。后文“新Base教师”均指这同一个
  回溯消融最佳模型，不是另一个独立模型。
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
├── p9_fresh_p8_new_teacher/           # P9：官方TinyViT + P8完整结构 + 新教师从头蒸馏
├── p10_adaptive_prompt_control/       # P10：验证闭环控制正提示频率
├── p11_pruned_p7_new_teacher/         # P11：删除P7高分支并按P9配置从头蒸馏
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
| P0历史教师输出缓存 | `../sam3_lightweight_stage3_distill_exp/cache/p0_teacher/` | 旧Base最终输出软目标，供P0～P8复用 |
| P9新教师输出缓存 | `p9_fresh_p8_new_teacher/cache/new_teacher_outputs/` | 来自Base回溯消融无域外负提示最佳模型，只含7个道路标线提示 |

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

以下结果来自同一批10张图片、阈值0.5。该批图片只有白实线和白虚线真值，因此其他类别只能用于观察误检，不能用于评估正样本IoU。该批图片实际全部来自同一无人机视频的相邻帧，只适合作为历史单场景回归集；其代表性限制见本节表后说明。两项验证损失填写的是该行实际接受10图测试的同一个checkpoint及同一轮次，不会用另一轮的最低loss替换任务权重。

| 模型 | 对应目录 | `val/supervised` | `val/loss` | 白实线IoU | 白实线Recall | 白虚线IoU | 白虚线Recall | 两类平均IoU |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| EfficientViT Stage-3 LoRA | `../sam3_lightweight_stage3_exp/` | 7.1777¹ | 7.1777 | 0.3905 | 0.5137 | 0.2153 | 0.2403 | 0.3029 |
| TinyViT P0 | 本目录根部 | 6.2367 | 10.0117 | 0.5187 | 0.6676 | 0.3400 | 0.4139 | 0.4293 |
| TinyViT P1图像特征KD | `p1_image_feature/` | 6.1320 | 10.2271 | 0.5343 | 0.6863 | 0.3490 | 0.4279 | 0.4417 |
| TinyViT P2 Stage 1/2/3 | `p2_image_stage123/` | 6.0514 | 10.1169 | 0.5454 | 0.7145 | 0.3652 | 0.4461 | 0.4553 |
| TinyViT P3全LoRA r16 | `p3_all_r16/` | 5.9835 | 10.0335 | 0.5487 | 0.7228 | 0.3639 | 0.4452 | 0.4563 |
| P4 Control任务最佳（epoch 0） | `p4_unfreeze_stage3_neck/` | 6.1161 | 10.1932 | 0.5411 | 0.7200 | 0.3680 | 0.4550 | 0.4546 |
| P4解冻任务最佳（epoch 1） | `p4_unfreeze_stage3_neck/` | 5.9318 | 9.9673 | 0.5180 | 0.6801 | 0.3664 | 0.4652 | 0.4422 |
| P5 Stage 2 DSConv（epoch 15） | `p5_dsconv_thin_line/` | 5.5815 | 9.5198 | 0.5746 | 0.7308 | 0.4967 | 0.5957 | 0.5357 |
| P6 Stage 1+2 DSConv（epoch 8） | `p6_multiscale_dsconv/` | 5.4088 | 9.3122 | 0.6214 | 0.7486 | 0.5610 | 0.6524 | 0.5912 |
| P7高/中分辨率FPN（epoch 9） | `p7_highres_fpn/` | 5.3814 | 9.2771 | 0.6319 | 0.7603 | 0.5687 | 0.6604 | 0.6003 |
| P8输入侧细线分支（epoch 4） | `p8_input_line_branch/` | 5.2508 | 9.1322 | 0.6772 | 0.7900 | 0.5960 | 0.6883 | 0.6366 |
| P9新教师从头蒸馏（epoch 19） | `p9_fresh_p8_new_teacher/` | 4.8604 | 8.5787 | 0.6284 | 0.6869 | 0.6595 | 0.7543 | 0.6439 |
| P10提示控制（epoch 4，验证最佳） | `p10_adaptive_prompt_control/` | 4.8032 | 8.5076 | 0.5766 | 0.6242 | 0.6649 | 0.7601 | 0.6207 |
| P10提示控制（epoch 9，最终） | `p10_adaptive_prompt_control/` | 4.8047 | 8.5136 | 0.5972 | 0.6502 | 0.6666 | 0.7607 | 0.6319 |
| P11删除P7高分支（epoch 19） | `p11_pruned_p7_new_teacher/` | 4.8085 | 8.5021 | 0.5959 | 0.6494 | 0.6182 | 0.7055 | 0.6071 |
| Base + DETR LoRA（epoch 4） | `../sam3_detr_exp/` | 4.3160¹ | 4.3160 | 0.7235 | 0.7883 | 0.6808 | 0.8226 | 0.7021 |
| 新Base教师（epoch 13；即Base回溯消融：无域外负提示） | `../sam3_detr_exp/negative_prompt_ablation/` | 4.1741¹ | 4.1741 | 0.7520 | 0.8608 | 0.7445 | 0.8423 | 0.7483 |

¹ 这三组不含蒸馏KD，历史日志没有另写`val/supervised`字段；表中以纯监督`val/loss`作为等价监督损失。P0～P11的`val/supervised`是训练代码直接记录的字段。由于各阶段加入的KD项和历史验证batch聚合口径并不完全相同，`val/loss`绝对值不能跨所有行直接排名；它主要用于同损失配置的相邻实验比较。P10相对P9的验证损失小幅下降但旧10图IoU没有超过P9；P11相对P9的验证损失同样下降，固定10图平均IoU和Recall却明显退化。这再次说明完整验证集训练目标、单视频10图和域外跨形态覆盖不是同一个指标。

P0到P1的实际变化：

- 白实线IoU相对提升约3.0%。
- 白虚线IoU相对提升约2.6%。
- 白虚线Recall相对提升约3.4%。
- 护栏负类误检数从10降到5；斑马线误检仍为10。

总体结论：TinyViT明显优于此前EfficientViT轻量基线，图像特征蒸馏和扩大图像LoRA覆盖范围带来稳定正收益；但P3把全部LoRA提高到r16后几乎没有继续提升，P4完整解冻Stage 3与neck后平均IoU反而下降，说明限制效果的主要因素既不是低秩容量，也不能通过直接放开高层视觉参数解决。轻量模型与Base仍有明显差距，尤其是白虚线Recall。

固定10图共有白实线268个实例、白虚线618个实例，实例数偏白虚线2.31倍；但真值像素分别为
2,108,174和584,226，面积又偏白实线3.61倍。更关键的是，10张图全部来自
`DJI_20251231162942_0002_V`的frame 001～037，相邻样本仅隔4帧。后续继续保留该集合用于历史
回归，但不能用它单独代表总体泛化或决定最终最佳模型。

## Loss比较注意事项

- Base的`val/loss`只包含SAM3监督损失。
- P0的`val/loss = supervised + 最终输出KD`。
- P1/P2/P3的`val/loss = supervised + 最终输出KD + 图像特征KD`。
- 不同实验的验证batch size曾经不同，原始损失还存在批量聚合尺度差异。
- 因此跨实验优先比较`val/supervised`、扣除新增KD后的趋势和统一实际IoU，不直接比较`val/loss`绝对值。

## 当前阶段结论与下一步

P2之后的结构实验目前把两类平均IoU从0.4553提高到P8 epoch 4的0.6366：

| 实验 | 训练起点 | 正式训练 | 实线IoU | 虚线IoU | 平均IoU | 结论 |
|---|---|---:|---:|---:|---:|---|
| P2 | P1最佳 | 10轮 | 0.5454 | 0.3652 | 0.4553 | Stage 1/2/3图像LoRA基线 |
| P5 | P2最佳 | 20轮 | 0.5746 | 0.4967 | 0.5357 | Stage 2细线旁路明显有效 |
| P6 | P5 epoch 15 | 10轮 | 0.6214 | 0.5610 | 0.5912 | Stage 1+2双尺度继续显著提升 |
| P7 | P6 epoch 8 | 10轮 | 0.6319 | 0.5687 | 0.6003 | 中分辨率直连小幅有效，高分辨率简单投影基本未采用 |
| P8（正式最佳） | P7 epoch 9 | 5轮，最佳epoch 4 | 0.6772 | 0.5960 | 0.6366 | 输入侧504分辨率分支持续有效，相对P7提高0.0363 |
| P9 | 官方TinyViT + P8完整结构 | 20轮，最佳epoch 19 | 0.6284 | 0.6595 | 0.6439 | 新教师从头蒸馏小幅超过P8，主要改善白虚线，`car`泛化恢复 |
| P10 | P9 epoch 19 | 10轮，验证最佳epoch 4；最终epoch 9 | 0.5972 | 0.6666 | 0.6319 | 最终轮优于普通续训但未超过P9；3张域外图恢复城市白实线 |
| P11 | 官方TinyViT + 删除P7高分支后的P8其余结构 | 20轮，最佳epoch 19 | 0.5959 | 0.6182 | 0.6071 | 验证loss低于P9，但平均IoU下降0.0369、虚线Recall下降0.0488，精简失败 |
| Base + DETR LoRA | Base权重 | 10轮内提前停止 | 0.7235 | 0.6808 | 0.7021 | 当前教师与效果上限 |
| 新Base教师（Base回溯消融无域外负提示模型） | 原始Base重新训练 | 20轮，最佳epoch 13 | 0.7520 | 0.7445 | 0.7483 | P9实际使用的教师与当前效果上限 |

P7相对P6平均IoU只提高0.0091，低于预设的0.02明确增益门槛；最终`gate_high=-0.0171`、`gate_mid=0.7620`，说明Stage 1方向特征简单插值到`288×288`没有被模型有效采用，Stage 2到`144×144`才是主要增益来源。

P8输入侧细线分支使用`PixelUnshuffle(2)`把RGB无损重排到504分辨率，在大幅下采样前运行低通道DSConv，再通过Stage 2语义门控融合到高分辨率FPN。5轮平均IoU依次为0.6155、0.6274、0.6340、0.6358、0.6366，epoch 4为正式任务最佳；相对P7提高0.0363，但仍低于Base + DETR LoRA的0.7021。代码已预留参数匹配的`1×9 + 9×1`普通长条卷积实现，因为严格对照尚未执行，目前结果不能证明动态蛇形采样本身优于普通长条卷积。

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

P8从P7 epoch 9开始，使用PixelUnshuffle把1008输入无损重排到504分辨率，在进一步下采样前用低通道横/纵DSConv提取细线，再通过Stage 2语义门控融合到`288×288`FPN。冻结P7全部已有参数，只训练新增分支5轮，实际使用四卡、每卡batch 1。epoch 4验证指标为`val/loss=9.1322`、`val/supervised=5.2508`；统一10图白实线IoU/Recall为0.6772/0.7900，白虚线为0.5960/0.6883，平均IoU为0.6366，相对P7提高0.0363。训练入口同时预留普通`1×9 + 9×1`长条卷积对照，详细逐轮结果、结构和命令见P8目录README。

### P9：官方TinyViT与P8完整结构使用新教师从头蒸馏

对应目录：`p9_fresh_p8_new_teacher/`

P9不继承任何P0～P8学生训练权重，从作者官方TinyViT Stage-3开始，同时挂载P5～P8完整细线结构，训练Stage 1/2/3图像LoRA、DETR LoRA、点积分类头、分割头和全部新增分支。教师切换为关闭域外纯负提示后平均IoU达到0.7483的Base权重；训练和教师缓存都显式关闭`person/dog/cat...`域外纯负提示，只保留7类道路标线内部负提示。

四卡20轮已完整结束，epoch 19为最低验证loss点：`val/loss=8.5787`、`val/supervised=4.8604`。
统一10图白实线IoU/Recall为0.6284/0.6869，白虚线为0.6595/0.7543，平均IoU为0.6439；
比P8提高0.0074，但仍比新Base教师低0.1043。`car`测试10图合计185个检测，首图15个且mask经
人工检查确实落在车辆上，证明新教师和关闭域外负提示的训练链路成功保留了开放类别能力。

### P10：验证闭环提示控制

对应目录：`p10_adaptive_prompt_control/`

P10以P9 epoch 19最佳权重为阶段2起点，取消此前未正式训练的整图均衡重采样方案，恢复原始图片随机采样。P10在现有验证前向中直接累计7类micro IoU、Precision和Recall，以`0.7×IoU+0.3×Recall`衡量各类验证质量，并平滑控制下一轮各类别正提示保留率；内部负提示始终保留，域外纯负提示继续关闭。

P10 epoch 0仍使用全部提示；Lightning先训练再验证，因此该轮不是未经训练的纯P9基线。白虚线
以目标虚实实例比2.0对应的基础率0.5327为先验，控制率限制在0.4～1.0，单轮最多变化0.1。

四卡已完整训练10轮。白虚线实际保留率由1.0降至epoch 9的0.5624，虚实训练实例比由3.75:1
降至约2.17:1。验证`control_score`最佳为epoch 4的0.7143，但旧10图平均IoU只有0.6207；最终
epoch 9的验证分数略低，旧10图反而恢复到0.6319。两者均未超过P9的0.6439，但epoch 9优于
普通P9续训的0.6256，说明提示控制小幅有效、仍不足以解决续训中的白实线退化。

3张无真值网络图上，P10 epoch 9检出白实线13个、白虚线29个；P9最佳分别为8和26，且P10
恢复了P9在城市多车道图中消失的白实线。这个结果与旧10图排序相反，进一步证明当前验证集、
旧单视频10图与跨形态域外图存在选模偏差。完整逐轮结果、固定10图限制与后续双路增强方案见
P10目录README。

### P11：删除P7高分支并使用新Base教师从头蒸馏

对应目录：`p11_pruned_p7_new_teacher/`

P11回到作者官方TinyViT Stage-3原始权重，不继承P9或P10学生权重。它保留P5、P6、P7中分辨率
分支和P8，只物理删除P7的Stage 1方向特征到`288×288` FPN高分支。该模块不会被创建，并非
简单把门控设为0。教师、无域外纯负提示数据配置、最终输出KD、三尺度图像特征KD、图像与
DETR LoRA、学习率、每卡batch 4和20轮计划均与P9一致；教师输出缓存可安全复用P9缓存。

删除后总参数为106,037,332、可训练参数为5,984,411，相对P9完整结构各减少33,281。四卡20轮
已完成，epoch 19的`val/supervised=4.8085`、`val/loss=8.5021`，均优于P9；但固定10图白实线
IoU/Recall只有0.5959/0.6494，白虚线为0.6182/0.7055，平均IoU 0.6071，比P9下降0.0369。
该实验未通过精简门槛，说明接近0的标量门控不足以证明路径无用；P7高分支继续保留。

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

P10验证闭环提示控制四卡训练：

```bash
bash sam3_lightweight_tinyvit_stage3_distill_exp/p10_adaptive_prompt_control/scripts/train_p10_adaptive.sh
```

P11删除P7高分支并从头蒸馏：

```bash
bash sam3_lightweight_tinyvit_stage3_distill_exp/p11_pruned_p7_new_teacher/scripts/train_p11_pruned_4gpu.sh
```
