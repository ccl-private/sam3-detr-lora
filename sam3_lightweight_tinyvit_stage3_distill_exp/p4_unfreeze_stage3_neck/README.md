# P4 解冻TinyViT Stage 3与neck实验

P4用于验证当前瓶颈是否来自冻结的视觉表征，而不是LoRA秩。实验从P2最佳权重建立两组同起点、同损失、同轮数对照。

## 两组实验

| 组别 | 图像侧 | DETR与输出头 | 用途 |
|---|---|---|---|
| Control | Stage 1/2/3继续r8 LoRA | DETR r8 LoRA，完整输出头 | 测量单纯续训3轮的收益 |
| Unfreeze | Stage 1/2保留r8 LoRA；Stage 3合并LoRA后完整解冻；neck完整解冻 | 与Control相同 | 测量完整视觉权重更新的独立收益 |

两组都保留P0最终输出KD和P1三尺度图像特征KD，并从同一个`../weights/p2_image_stage123_r8.best.pt`开始。

## 学习率

- TinyViT Stage 3完整参数：`2e-6`
- FPN neck完整参数：`1e-5`
- Stage 1/2图像LoRA：`1e-5`
- DETR LoRA：`5e-5`
- 点积分类头和分割头：`2e-5`

## 执行顺序

先分别完成两组单卡冒烟测试。正式训练时两组都运行3轮：

```bash
bash sam3_lightweight_tinyvit_stage3_distill_exp/p4_unfreeze_stage3_neck/scripts/train_p4_control.sh
bash sam3_lightweight_tinyvit_stage3_distill_exp/p4_unfreeze_stage3_neck/scripts/train_p4_unfreeze.sh
```

不要同时启动两条默认四卡脚本。完成一组后再启动另一组，保证GPU数量和全局batch一致。

## 输出

```text
../weights/p4_control_p2_continue.best.pt
../weights/p4_unfreeze_stage3_neck.best.pt
../logs/p4_control_p2_continue/
../logs/p4_unfreeze_stage3_neck/
```

最终使用相同10张图片、7个提示和0.5阈值比较两组。只有Unfreeze相对Control平均IoU提高至少0.01，或白虚线Recall提高至少0.02且负类误检不明显增加，才认为完整解冻有效。

两条正式脚本都启用`--save-every-epoch`，会额外保存`.epoch0.pt`、`.epoch1.pt`和`.epoch2.pt`。训练后逐个运行统一10图评测，按平均IoU、白虚线Recall和负类误检共同选模；`.best.pt`仍只是最低验证loss权重，不能直接当作最终任务最佳权重。

## 冒烟验证结果

两组都已完成1个训练step和1个验证step：

- Control：总参数104,689,850，可训练参数4,636,929，训练与验证通过。
- Unfreeze：总参数104,575,162，可训练参数17,164,125。
- Unfreeze中完整Stage 3参数4,839,772，完整neck参数7,802,112。
- 成功把Stage 3中的8个r8 LoRA模块合并进基础权重并移除参数化；Stage 1/2图像LoRA继续保留。
- 梯度范数：DETR LoRA 242.31、Stage 1/2图像LoRA 227.28、完整Stage 3 247.34、完整neck 766.12、输出头137.52，五组均有有效梯度。
- 专用checkpoint已验证包含38个Stage 3张量和22个neck张量，Stage 3不再包含LoRA参数化键。

冒烟验证只确认工程链路，不比较两组loss高低。正式结果以完整3轮训练和统一10图评测为准。

## 正式训练与统一评测结果

Control和Unfreeze都按计划完整训练3轮（epoch 0～2），并逐轮使用相同10张图片、7个提示和0.5阈值评测。下表分别列出两组任务指标最好的轮次；“最佳轮次”表示从全部3轮中按实际任务指标选出，并非提前停止。

| 组别 | 最佳轮次 | 白实线IoU | 白实线Recall | 白虚线IoU | 白虚线Recall | 两类平均IoU |
|---|---:|---:|---:|---:|---:|---:|
| Control | epoch 0 | 0.5411 | 0.7200 | 0.3680 | 0.4550 | **0.4546** |
| Unfreeze | epoch 1 | 0.5180 | 0.6801 | 0.3664 | 0.4652 | **0.4422** |

Unfreeze最后一轮epoch 2的平均IoU进一步降至0.4395。相对Control最佳轮次，Unfreeze最佳轮次平均IoU下降0.0124，白实线IoU下降0.0231，白实线Recall下降0.0399；白虚线Recall仅提高0.0102，未达到预设的0.02门槛。两组斑马线负类检测数均约10，lane barrier均为0，没有观察到明显新增误检。

训练目标与任务指标出现背离：Unfreeze最低验证loss为9.9673，低于Control的10.1254，图像特征KD也由约0.3301降至0.3058，但实际平均IoU更低。这说明完整解冻Stage 3与neck更容易拟合当前监督和蒸馏目标，却没有转化成更好的道路标线分割能力。

最终结论：**P4无效**。它没有满足任一通过条件，不继续追加训练轮数；TinyViT主线仍以P2作为更稳妥的轻量基线。评测汇总位于`../tests/output/p4_*_epoch*_first10_threshold_05/summary.json`，该测试输出不提交Git。
