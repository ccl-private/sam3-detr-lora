# 域外负提示消融实验

本目录只保存Base DETR LoRA跨类别泛化退化的负提示消融实验。它属于`sam3_detr_exp`，不新建
顶层轻量化实验目录。训练脚本、配置和说明提交Git；`weights/`、`logs/`和`tests/output/`不提交。

## 已完成实验：关闭域外通用负提示

目标是复现`roadline_r8_a16_lr2e4`的正式训练条件，只把每张图片额外采样的
`person/dog/cat...`域外纯负提示数量从2改为0。以下条件保持不变：

- 仍使用`multi_prompt`，每张图仍展开全部7个道路标线类别；
- 图中不存在的道路标线类别仍作为数据集内负提示；
- SAM3原生分类、Presence、框、mask、辅助层和O2M监督保持不变；
- DETR Encoder/Decoder r8 LoRA，`alpha=16`、`dropout=0.05`；
- 完整训练点积分类头和分割头，学习率`2e-4`；
- 4卡、每卡图片batch 2，完整训练20轮。

### 与后续实验的身份关系

本实验的最佳模型
`weights/roadline_r8_a16_lr2e4_no_generic_negatives.best.pt`就是后续文档统一称呼的
**“新Base教师”**。它不是额外下载或另行训练的模型：它就是“Base回溯消融：无域外负提示”
本身的产物。TinyViT P9使用该权重缓存教师输出并完成20轮从头蒸馏。

训练入口新增`--num-generic-negatives`覆盖参数。不传时继续使用YAML中的
`prompt_training.num_negatives`；传0时关闭训练阶段域外负提示。验证阶段原本就不随机加入域外
负提示。

## 执行命令

在仓库根目录执行：

```bash
bash sam3_detr_exp/negative_prompt_ablation/scripts/train_no_generic_negatives.sh
```

正式输出：

```text
weights/roadline_r8_a16_lr2e4_no_generic_negatives.pt
weights/roadline_r8_a16_lr2e4_no_generic_negatives.best.pt
logs/lightning_logs/version_*/metrics.csv
```

启动日志必须显示：

```text
mode=multi_prompt num_generic_negatives=0
```

## 对照关系与判定

| 权重 | 提示构造 | 已知`car`结果 |
|---|---|---:|
| `../weights_lora/roadline_sam3_loss_lora.best.pt` | 单正提示、SAM3原生loss/Presence | 20 |
| `../weights_lora/roadline_r8_a16_lr2e4.best.pt` | 内部负提示 + 每图2个域外负提示 | 0 |
| 本实验最佳权重 | 只保留内部负提示 | 20 |

## 训练与测试结果

`logs/lightning_logs/version_0/metrics.csv`记录了epoch 0～19共20轮完整验证。最佳点为epoch 13：

| 指标 | 数值 |
|---|---:|
| `val/loss` | 4.1741 |
| `val/loss_ce` | 0.0854 |
| `val/presence_loss` | 0.0468 |
| 最后一轮`val/loss` | 4.3106 |

历史上同为20轮、每图加入2个域外负提示的`roadline_multi_prompt_lora.best.pt`最低
`val/loss=4.2201`。关闭域外负提示后反而降低0.0460，因此域外纯负提示不是道路标线收敛所必需。

统一10图、7个道路标线提示、阈值0.5的实际结果：

| 类别 | IoU | Precision | Recall | 检测数 |
|---|---:|---:|---:|---:|
| 白实线 | 0.7520 | 0.8561 | 0.8608 | 185 |
| 白虚线 | 0.7445 | 0.8651 | 0.8423 | 534 |
| 两类平均IoU | 0.7483 | - | - | - |

旧正式教师`roadline_r8_a16_lr2e4.best.pt`对应实线/虚线IoU为0.7235/0.6808，平均0.7021。
新模型平均IoU提高0.0462，说明关闭域外负提示没有牺牲专项能力，反而得到更好结果。

跨类别测试固定使用`DJI_20251231162942_0002_V_frame_001.png`、提示`car`和阈值0.5：

| 权重 | `car`检测数 |
|---|---:|
| 早期单正提示SAM3-loss模型 | 20 |
| 含域外负提示的旧正式教师 | 0 |
| 本实验无域外负提示最佳模型 | 20 |

### 实验结论

在保留全部7个道路标线提示和数据集内部空目标负提示的情况下，仅关闭没有正样本配对的域外通用
负提示，就同时实现了更低验证loss、更高道路标线IoU，并把`car`检测从0恢复到20。当前证据直接
支持：跨类别泛化消失的主要原因是把`person/dog/cat...`长期作为纯负提示，而不是多提示机制、
道路标线内部负提示或SAM3原生Presence loss本身。

该结论目前由`car`一个域外类别验证。后续仍应补测`person`、`dog`以及从未出现在域外负词表中的
类别，确认恢复是否具有普遍性；在此之前，所有正式训练默认保持`--num-generic-negatives 0`。
