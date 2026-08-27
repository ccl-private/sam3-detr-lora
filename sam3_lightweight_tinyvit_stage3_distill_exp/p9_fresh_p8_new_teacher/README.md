# P9 原始TinyViT与P8完整结构新教师蒸馏

## 实验目标

本实验不继承P0～P8任何已经训练过的学生LoRA、输出头或细线分支权重。从作者官方
TinyViT Stage-3开始，一次性挂载P5～P8完整细线结构，使用关闭域外纯负提示后重新训练得到的
Base教师完整蒸馏20轮。

学生保留7个道路标线类别之间的数据集内负提示、SAM3 Presence、分类、框、mask、辅助层与O2M
监督；训练和教师缓存均不加入`person`、`dog`、`cat`等域外纯负提示。

## 初始化与可训练模块

- 学生基模：`sam3_lightweight_stage3_exp/input/efficientsam3_tinyvit_stage3.pt`。
- 历史学生权重：不加载。
- 教师：`sam3_detr_exp/negative_prompt_ablation/weights/roadline_r8_a16_lr2e4_no_generic_negatives.best.pt`。
- 图像LoRA：TinyViT Stage 1/2/3，r8、alpha16。
- DETR LoRA：Encoder/Decoder，r8、alpha16。
- 完整训练点积分类头和分割头。
- 同时训练P5 Stage 2 DSConv、P6 Stage 1 DSConv、P7 FPN直连和P8输入侧504分辨率分支。
- 所有新增残差门控从0初始化，初始输出严格等价于官方TinyViT；门控更新后分支内部开始获得梯度。

## 第一步：缓存新教师输出

在仓库根目录执行：

```bash
bash sam3_lightweight_tinyvit_stage3_distill_exp/p9_fresh_p8_new_teacher/scripts/cache_new_teacher_outputs_4gpu.sh
```

四个进程分别写入同一个`cache/new_teacher_outputs/`。脚本显式使用
`--exclude-generic-prompts`，缓存中只有7个道路标线提示。图像特征教师仍是冻结的原始Base视觉
骨干，因此复用已经完成的`cache/p1_image_features/`。

检查缓存进度：

```bash
find sam3_lightweight_tinyvit_stage3_distill_exp/p9_fresh_p8_new_teacher/cache/new_teacher_outputs/train -name '*.pt' | wc -l
find sam3_lightweight_tinyvit_stage3_distill_exp/p9_fresh_p8_new_teacher/cache/new_teacher_outputs/val -name '*.pt' | wc -l
```

完整数量应为训练9351、验证2343。

## 第二步：四卡训练20轮

缓存完成后执行：

```bash
bash sam3_lightweight_tinyvit_stage3_distill_exp/p9_fresh_p8_new_teacher/scripts/train_p9_fresh_p8_4gpu.sh
```

默认四卡、每卡图片batch 4、全局batch 16，保存每一轮权重。该配置与此前TinyViT图像LoRA
训练使用的每卡batch 4一致，并已通过单卡batch 4的完整前向、反向和验证冒烟；如果正式数据出现
异常峰值导致显存不足，再降到每卡2。启动日志必须包含：

```text
P9起点=官方TinyViT Stage-3，域外负提示=0
```

输出位置：

- 日志：`sam3_lightweight_tinyvit_stage3_distill_exp/logs/p9_fresh_p8_new_teacher/`
- 每轮权重：`sam3_lightweight_tinyvit_stage3_distill_exp/weights/p9_fresh_p8_new_teacher.epochN.pt`
- 最低验证loss权重：`sam3_lightweight_tinyvit_stage3_distill_exp/weights/p9_fresh_p8_new_teacher.best.pt`

正式选模以逐轮统一实际IoU为准，不以不同损失项混合后的`val/loss`单独决定。

## 正式训练与测试结果

四卡训练已完整完成20轮（epoch 0～19），没有提前停止。验证指标在epoch 19达到最低点，
`p9_fresh_p8_new_teacher.best.pt`与epoch 19对应：

| 指标 | epoch 0 | epoch 19（最佳） |
|---|---:|---:|
| `val/loss` | 11.6501 | 8.5787 |
| `val/supervised` | 7.1849 | 4.8604 |

使用相同10张道路图片、全部7个道路标线提示和0.5阈值评测最佳权重：

| 模型 | 白实线IoU | 白实线Precision | 白实线Recall | 白虚线IoU | 白虚线Precision | 白虚线Recall | 平均IoU |
|---|---:|---:|---:|---:|---:|---:|---:|
| P8 epoch 4 | 0.6772 | 0.8258 | 0.7900 | 0.5960 | 0.8163 | 0.6883 | 0.6366 |
| P9 epoch 19 | 0.6284 | 0.8805 | 0.6869 | 0.6595 | 0.8400 | 0.7543 | 0.6439 |
| 新Base教师 | 0.7520 | 0.8561 | 0.8608 | 0.7445 | 0.8651 | 0.8423 | 0.7483 |

P9相对P8平均IoU提高0.0074，收益主要来自白虚线IoU提高0.0635；白实线IoU反而下降0.0488。
因此从官方TinyViT和新教师重新蒸馏有效但增益很小，没有达到0.68验收线；相对新Base教师仍差
0.1043。评测结果位于`../tests/output/p9_fresh_p8_best_first10_threshold_05/`，该目录不提交Git。

### `car`开放类别泛化

使用同一批10张道路图、提示`car`和阈值0.5测试P9最佳权重，每张图检测数依次为
`15/20/20/21/20/17/16/18/19/19`，合计185个；固定首图检测15个。人工检查首图mask确认预测
落在真实车辆上，并非道路标线误检。新Base教师在首图检测20个，说明P9尚未完全追平教师，但
已经明确保留开放类别能力，没有重现P2～P8的`car=0`退化。结果位于
`../tests/output/car_p9_fresh_p8_best_first10_threshold_05/`。由于测试图片没有`car`人工真值，
可视化中的红色只表示“无真值可匹配”，不能解释为人工确认的误检。

## 验收结论

- 已超过P8平均IoU 0.6366，但只提高到0.6439，属于小幅有效。
- 未达到0.68明确收益线，也未接近0.70。
- `car`已确认保留；`person`和`dog`仍待测试。
- 10图中`zebra crossing`产生8个无真值候选，P8为11个；候选数没有增加，但无该类真值，
  不能据此计算准确率。
