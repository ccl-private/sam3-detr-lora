# P11 删除P7高分支并使用新Base教师从头蒸馏

P11是P9的结构精简对照。学生从作者官方TinyViT Stage-3原始权重开始，不加载P8、P9或P10学生权重；教师继续使用关闭域外纯负提示后重新训练得到的新Base教师。

## 实验变量

P9完整结构为：

```text
P5 Stage 2低分辨率DSConv
P6 Stage 1低分辨率DSConv
P7 Stage 1 → 288×288高分辨率FPN
P7 Stage 2 → 144×144中分辨率FPN
P8 输入侧504分辨率细线分支
```

P11只删除：

```text
P7 Stage 1方向特征 → 288×288 FPN[0]
```

保留P5、P6、P7中分辨率分支和P8。训练损失、LoRA覆盖、学习率、batch、训练轮数、数据划分、提示词和教师均与P9一致。代码不会创建`p7_highres_fpn_adapters.high`模块，不是简单把门控固定为0。

删除依据是该高分支在P8、P9和P10 epoch 9中的门控绝对值分别只有0.0171、0.0174和0.0176，连续三次训练都接近0；P7中分辨率门控在P10 epoch 9达到1.376，因此继续保留。

## 教师与缓存

正式教师权重：

```text
sam3_detr_exp/negative_prompt_ablation/weights/roadline_r8_a16_lr2e4_no_generic_negatives.best.pt
```

P11复用P9的新教师输出缓存：

```text
sam3_lightweight_tinyvit_stage3_distill_exp/p9_fresh_p8_new_teacher/cache/new_teacher_outputs
```

该缓存只包含教师对相同训练集、验证集和7个道路标线提示的输出。P11与P9的教师、数据划分、提示和域外负提示设置完全一致，因此不重复生成缓存。图像特征KD继续复用公共P1教师特征缓存。

## 正式训练

```bash
bash sam3_lightweight_tinyvit_stage3_distill_exp/p11_pruned_p7_new_teacher/scripts/train_p11_pruned_4gpu.sh
```

默认四卡、每卡batch 4、20轮、逐轮保存：

- 最佳验证损失权重：`weights/p11_pruned_p7_new_teacher.best.pt`
- 最终权重：`weights/p11_pruned_p7_new_teacher.pt`
- 逐轮权重：`weights/p11_pruned_p7_new_teacher.epochN.pt`
- 日志：`logs/p11_pruned_p7_new_teacher/`

## 判定方法

P11是对P9的单变量结构消融。训练完成后至少比较：

- 完整验证集`val/supervised`和`val/loss`；
- 历史固定10图的白实线、白虚线IoU和Recall；
- 3张网络图的跨形态候选覆盖；
- 单图推理延迟与峰值显存。

如果P11相对P9平均IoU下降不超过0.005、白虚线Recall下降不超过0.01，同时延迟或显存有可测量改善，则正式删除P7高分支。固定10图只作为历史单视频回归口径，不单独决定最终结论。

## 实现与冒烟验证

当前实现已通过以下检查：

- 模型中不存在`p7_highres_fpn_adapters.high`，只存在`mid`适配器；
- 总参数106,037,332、可训练参数5,984,411，相对P9完整结构均减少33,281；
- GPU 0完成两个训练batch和一个验证batch，step 0与step 1中DETR LoRA、图像LoRA、输出头、P5、P6、P7中分辨率和P8梯度均非零；
- 冒烟checkpoint保存了`p7_high_branch_enabled: false`元数据；
- 统一评测器能根据元数据重建精简结构、严格加载权重并完成单图7提示推理。

冒烟权重只训练两个batch，其预测数值不用于效果比较。

## 正式训练与评测结果

四卡20轮已完整结束，`val/loss`最低点为epoch 19，`val/supervised`最低点为epoch 17；两轮监督损失只差0.0001，因此正式使用按总损失保存的epoch 19权重：

| 指标 | P9 epoch 19 | P11 epoch 19 | 变化 |
|---|---:|---:|---:|
| `val/supervised` | 4.8604 | **4.8085** | -0.0519 |
| `val/kd` | 3.7183 | **3.6935** | -0.0248 |
| `val/loss` | 8.5787 | **8.5021** | -0.0766 |

使用相同固定10图、7提示和0.5阈值评测正式最佳权重：

| 指标 | P9 epoch 19 | P11 epoch 19 | 变化 |
|---|---:|---:|---:|
| 白实线IoU | **0.6284** | 0.5959 | -0.0325 |
| 白实线Recall | **0.6869** | 0.6494 | -0.0375 |
| 白虚线IoU | **0.6595** | 0.6182 | -0.0413 |
| 白虚线Recall | **0.7543** | 0.7055 | -0.0488 |
| 两类平均IoU | **0.6439** | 0.6071 | -0.0369 |

P11白实线Precision为0.8786、白虚线Precision为0.8332，与P9接近；主要退化来自Recall下降。固定10图结果位于`../tests/output/p11_pruned_p7_best_first10_threshold_05/`，该输出目录不提交Git。

最终门控为：P5 `-0.7198`、P6 `-0.8560`、P7中分辨率`1.3589`、P8 `0.1229`。checkpoint中P7高分支张量数量为0，确认评测使用的确实是物理精简结构。

## 正式结论

P11未通过预设精简标准。虽然完整验证集监督损失下降约1.1%、总损失下降约0.9%，固定10图平均IoU却下降0.0369，白虚线Recall下降0.0488，明显超过允许下降0.005和0.01的门槛。删除该分支又只减少33,281个参数，FP16约65 KiB，预期端到端加速有限。

因此不能根据门控标量接近0直接认定分支无用。门控前特征幅值和该路径对联合优化的影响同样重要。当前正式部署结构继续保留P7高分支；P11作为“降低验证loss但损害任务Recall”的失败消融保留，不作为后续学生起点。
