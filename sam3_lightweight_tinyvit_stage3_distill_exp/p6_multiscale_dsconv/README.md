# P6 Stage 1 + Stage 2双尺度DSConv实验

## 目标与起点

P6直接从P5的实际任务最佳权重`weights/p5a_dsconv_frozen_20ep.epoch15.pt`继续。P5已经把白实线/白虚线平均IoU从P2的0.4553提高到0.5357，证明高分辨率DSConv分支有效；本阶段跳过普通卷积消融，进一步利用更高分辨率的TinyViT Stage 1特征。

P6的训练起点必须严格保持P5 epoch 15输出：

1. 恢复P5权重中的图像LoRA、DETR LoRA和输出头。
2. 挂载并恢复已经训练好的Stage 2 DSConv分支。
3. 新增Stage 1 DSConv分支，其融合门控初始化为0。
4. 冻结基础模型、全部LoRA、输出头和Stage 2分支，只训练Stage 1分支。

因此新增分支在第一个优化步骤之前不改变P5预测，后续增益可以归因到Stage 1分支。

## 结构与训练范围

```text
TinyViT Stage 1特征 ── 新增DSConv（可训练，64通道） ── 零门控 ┐
TinyViT Stage 2特征 ── P5 DSConv（冻结，128通道）───────────┼─ 学生投影/FPN ─ DETR
TinyViT原输出 ──────────────────────────────────────────────┘
```

Stage 1分辨率高于Stage 2，动态采样的显存成本更高，因此第一版把分支中间通道从128降为64，并把每卡batch从4降为2。若实测显存仍有余量，再提高batch；不要首先减少提示词或训练样本。

损失保持P5一致：真实标签监督、最终输出蒸馏和三尺度图像特征蒸馏均保留，不同时引入新损失。

## P6门控机制

P6同时保留P5已经训练好的Stage 2门控，并为新增Stage 1分支设置一个独立门控。两级残差都先融合到学生图像编码器的`1024×72×72`输出：

```text
F_stage2 = gate_stage2 × Branch_stage2(Stage2特征)
F_stage1 = gate_stage1 × Branch_stage1(Stage1特征)

F_out = F_tinyvit + F_stage2 + F_stage1
```

其中：

- `gate_stage2`来自P5 epoch 15，加载值为`-1.24412823`，与P5分支一起冻结；它不能重新初始化为0，否则P6起点会丢失P5已获得的能力。
- `gate_stage1`属于P6新增分支，初始化为0并参与训练；因此刚挂载P6时，新增Stage 1残差为0，模型输出严格等价于P5 epoch 15。

门控是无约束的可学习标量。正值表示按当前投影方向加上分支残差，负值表示减去残差；由于分支投影权重本身也可以改变符号，门控正负不能独立解释为分支有益或有害，应主要结合绝对值、稳定性和实际IoU判断。

零门控会产生预期的分阶段梯度行为：

1. 第一个优化step中，Stage 1门控可以获得梯度。
2. Stage 1分支主体被零门控截断，第一个step的偏移、投影和融合参数梯度为0。
3. 门控离开0后，从第二个step开始，Stage 1分支主体获得梯度。

这是一种无扰动初始化策略，不是梯度异常。它优先保证P6从已验证的P5能力起步，再逐渐学习是否以及以多大强度使用新增Stage 1特征。

门控数值的基本解释：

- `|gate|`长期接近0：模型整体上基本没有采用该分支。
- `|gate|`持续增长后稳定：该分支持续影响融合特征。
- 门控增大但实际IoU不升：不能认定该分支有效。
- 门控剧烈振荡：可能存在学习率过高或梯度冲突。

P6每个分支只使用一个标量控制全部输出通道。这种方式参数少、起点安全，适合先验证整个分支是否具有总体价值；它不能选择单独的有效通道，只有标量门控被证明确实限制效果后，才考虑通道级门控。

## 执行训练

默认四卡、10轮、逐轮保存：

```bash
bash sam3_lightweight_tinyvit_stage3_distill_exp/p6_multiscale_dsconv/scripts/train_p6_stage1_frozen_p5.sh
```

输出位置：

- 日志：`sam3_lightweight_tinyvit_stage3_distill_exp/logs/p6_stage1_frozen_p5/`
- 最佳权重：`sam3_lightweight_tinyvit_stage3_distill_exp/weights/p6_stage1_frozen_p5.best.pt`
- 每轮权重：`sam3_lightweight_tinyvit_stage3_distill_exp/weights/p6_stage1_frozen_p5.epochN.pt`

## 验证要求

正式训练前必须通过：

- P5 epoch 15与P6零门控初始化的FPN输出一致性检查。
- 只有`p6_stage1_thin_line_branch.*`参数可训练。
- Stage 2门控和权重加载值与P5 checkpoint一致。
- 两个训练step后Stage 1门控、偏移、投影和融合参数均获得梯度。
- checkpoint可由统一评测脚本自动挂载两个分支。

正式评测继续使用相同10图、7提示词、阈值0.5。首要指标是白虚线IoU能否从P5的0.4967继续提高；平均IoU至少达到0.5557才认为新增Stage 1具有明确价值。若平均IoU提升不足0.01，或负类误检明显增加，则不进入双分支联合微调。

## 已完成验证

- Stage 1实际输入为`(1, 128, 126, 126)`，Stage 2输入为`(1, 256, 63, 63)`，两级分支均对齐到`(1, 1024, 72, 72)`。
- 总参数105,887,456；P5冻结分支896,019；P6新增且可训练参数301,587。
- Stage 1门控为0时，P6输出与P5 epoch 15逐元素相等，最大绝对误差0。
- P5 Stage 2门控恢复前后均为`-1.24412823`。
- 两batch训练冒烟通过：首步仅门控有梯度，第二步偏移、投影、融合和门控梯度均非零。
- 冒烟checkpoint包含P5与P6各14个分支张量，统一评测脚本成功加载并完成单图7提示词推理。

## 文件

| 文件 | 用途 |
|---|---|
| `multiscale_dsconv.py` | 恢复P5、挂载Stage 1分支及P6 checkpoint加载 |
| `train_p6_multiscale.py` | 冻结P5、仅训练Stage 1分支的训练入口 |
| `configs/p6_stage1_frozen_p5.yaml` | 实验结构、冻结范围和超参数记录 |
| `scripts/train_p6_stage1_frozen_p5.sh` | 默认四卡训练脚本 |

## 正式结果与结论

P6从P5 epoch 15开始完整训练10轮，统一10图实际IoU最佳为epoch 8，而最低验证loss出现在epoch 9。最终按任务指标选择`weights/p6_stage1_frozen_p5.epoch8.pt`：

| 指标 | P5 epoch 15 | P6 epoch 8 | 变化 |
|---|---:|---:|---:|
| 白实线IoU | 0.5746 | 0.6214 | +0.0467 |
| 白虚线IoU | 0.4967 | 0.5610 | +0.0643 |
| 两类平均IoU | 0.5357 | 0.5912 | +0.0555 |
| 白虚线Recall | 0.5957 | 0.6524 | +0.0567 |

epoch 8验证指标为`val/loss=9.3122`、`val/supervised=5.4088`；斑马线误检11、护栏误检0。结果超过预设0.5557验收线，证明在冻结P5的条件下增加Stage 1方向分支具有明确增量。epoch 8～9实际IoU和验证loss均进入平台，不追加到20轮。
