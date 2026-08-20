# SAM3 道路标线轻量化实验总览

本仓库围绕SAM3道路标线文本提示分割，依次验证Base DETR LoRA、早期TinyViT轻量化、EfficientViT Stage-3、EfficientViT蒸馏和TinyViT Stage-3蒸馏。

根README只负责实验导航和横向结论。每个实验的模型结构、下载方式、训练命令、测试配置、权重与完整指标均记录在对应的`*exp/README.md`中。实验代码和产物只应放在所属实验目录内。

## 实验目录

| 实验目录 | 实验内容 | 当前作用 | 详细文档 |
|---|---|---|---|
| `sam3_detr_exp/` | SAM3 Base模块化、DETR LoRA道路标线训练 | 效果上限与蒸馏教师 | [进入实验](sam3_detr_exp/README.md) |
| `sam3_lightweight_exp/` | 早期TinyViT-S固定词表与DETR LoRA | 第一代轻量化探索 | [进入实验](sam3_lightweight_exp/README.md) |
| `sam3_lightweight_stage3_exp/` | 官方EfficientViT Stage-3 EV-M评测与LoRA | 官方Stage-3轻量基线 | [进入实验](sam3_lightweight_stage3_exp/README.md) |
| `sam3_lightweight_stage3_distill_exp/` | Base向EfficientViT Stage-3进行P0输出蒸馏 | EfficientViT蒸馏路线 | [进入实验](sam3_lightweight_stage3_distill_exp/README.md) |
| `sam3_lightweight_tinyvit_stage3_distill_exp/` | Base向TinyViT Stage-3进行输出和图像特征蒸馏 | 当前轻量化主实验 | [进入实验](sam3_lightweight_tinyvit_stage3_distill_exp/README.md) |

## 实验关系

```text
SAM3 Base
└── sam3_detr_exp：DETR LoRA，作为教师和效果上限
    ├── sam3_lightweight_stage3_distill_exp：蒸馏到EfficientViT Stage-3
    └── sam3_lightweight_tinyvit_stage3_distill_exp：蒸馏到TinyViT Stage-3

sam3_lightweight_exp：早期TinyViT-S固定词表实验
sam3_lightweight_stage3_exp：官方EfficientViT Stage-3直接LoRA基线
```

## 各实验训练基础、进度与量化结论

“完成轮数”按实际完成验证的epoch计数，不把中途退出的epoch算作完整一轮。“最佳轮次”保留日志中的零基编号。

| 实验及所属目录 | 在什么基础上训练 | 本次训练改动 | 实际轮数与停止情况 | 统一10图结果（阈值0.5） | 实验结论 |
|---|---|---|---|---|---|
| [Base + DETR LoRA（`sam3_detr_exp/`）](sam3_detr_exp/README.md) | 原始SAM3 Base模块化权重 | DETR Encoder/Decoder r8 LoRA，并训练点积分类头和分割头 | 日志`lightning_logs/version_18`完成8轮验证（epoch 0～7），epoch 8未完成；最佳验证loss在epoch 4。现有文档未记录提前停止原因 | 白实线IoU 0.7235、Recall 0.7883；白虚线IoU 0.6808、Recall 0.8226；平均IoU 0.7021 | 当前教师和效果上限，但训练停止原因需要补录 |
| [早期TinyViT-S（`sam3_lightweight_exp/`）](sam3_lightweight_exp/README.md) | `efficient_sam3_tinyvit_s.pt`，文本编码器替换为固定词表 | DETR r8 LoRA，并训练点积分类头和分割头 | 原计划3轮，只完成epoch 0后主动停止，用于先验证可行性；`val/loss=9.3882` | 未计算IoU/Recall；只记录白实线77个、白虚线49个检测及平均置信度0.5956/0.6141 | 只能证明训练链路与领域响应可行，不能与完整实验作效果排名 |
| [EfficientViT Stage-3 LoRA（`sam3_lightweight_stage3_exp/`）](sam3_lightweight_stage3_exp/README.md) | 官方Stage-3 EV-M：EfficientViT-B1 + MobileCLIP-S0 | 图像与文本编码器冻结；DETR r8 LoRA，并训练点积分类头和分割头 | 完成计划的20轮（epoch 0～19），最佳验证loss为epoch 10的7.1777；没有提前停止 | 白实线IoU 0.3905、Recall 0.5137；白虚线IoU 0.2153、Recall 0.2403；平均IoU 0.3029 | 完整训练后仍明显落后Base，尤其是白虚线召回 |
| [EfficientViT Stage-3 P0蒸馏（`sam3_lightweight_stage3_distill_exp/`）](sam3_lightweight_stage3_distill_exp/README.md) | 上一行Stage-3最佳LoRA；教师为Base + DETR最佳LoRA | 保留真实监督，增加最终层分类、presence、框和mask KD；学生仍训练r8 LoRA与两个输出头 | 计划10轮；日志`version_5`完成8轮验证（epoch 0～7），epoch 8未完成后主动停止；最低`val/loss=11.1575`在epoch 1 | 目录内未保留可复核的统一10图IoU/Recall结果 | 只能确认蒸馏训练完成过且验证loss未持续改善；没有统一实测数值，不能量化其相对Stage-3 LoRA的收益 |
| [TinyViT Stage-3 P0（实验根目录）](sam3_lightweight_tinyvit_stage3_distill_exp/README.md) | 官方TinyViT Stage-3基模；DETR和输出头初始化自Stage-3学生LoRA；教师为Base最佳LoRA | 图像Stage 2/3 r8 LoRA + DETR r8 LoRA + 输出头；增加最终输出KD | 计划10轮，完成7轮（epoch 0～6）后主动停止；最佳点epoch 6，`val/loss=10.0117` | 白实线IoU 0.5187、Recall 0.6676；白虚线IoU 0.3400、Recall 0.4139；平均IoU 0.4293 | 明显超过EfficientViT Stage-3，但仍远低于Base |
| [TinyViT Stage-3 P1（`p1_image_feature/`）](sam3_lightweight_tinyvit_stage3_distill_exp/p1_image_feature/README.md) | 从P0最佳权重继续训练 | 保留P0全部损失，新增三尺度图像FPN特征KD；图像LoRA仍为Stage 2/3 r8 | 计划10轮，完成7轮（epoch 0～6）后主动停止；最佳点epoch 6，`val/loss=10.2271` | 白实线IoU 0.5343、Recall 0.6863；白虚线IoU 0.3490、Recall 0.4279；平均IoU 0.4417 | 相对P0平均IoU +0.0124，特征蒸馏有效但增益较小 |
| [TinyViT Stage-3 P2（`p2_image_stage123/`）](sam3_lightweight_tinyvit_stage3_distill_exp/p2_image_stage123/README.md) | 从P1最佳权重继续训练 | 在相同蒸馏配置上，把图像r8 LoRA从Stage 2/3扩展到Stage 1/2/3 | 计划10轮，完成6轮（epoch 0～5）后主动停止；最佳点epoch 5，`val/loss=10.1169` | 白实线IoU 0.5454、Recall 0.7145；白虚线IoU 0.3652、Recall 0.4461；平均IoU 0.4553 | 当前最佳轻量结果；相对P1平均IoU +0.0136，但收益同时混有额外训练轮数，不能完全归因于Stage 1 LoRA |
| [TinyViT Stage-3 P3（`p3_image_r16/`）](sam3_lightweight_tinyvit_stage3_distill_exp/p3_image_r16/README.md) | 将P2最佳图像r8增量合并进基权重，再挂载零增量图像r16；DETR r8和输出头继承P2 | 只把图像LoRA从r8提高到r16，其他配置不变 | 正式训练尚未开始；仅完成1个训练step和1个验证step的冒烟测试 | 无正式结果；转换一致性单图白实线IoU 0.5373→0.5375、白虚线0.3978→0.4012 | 目前只能证明转换与训练链路正确，不能判断r16是否提升效果 |

统一数值均以相同10张道路图片、相同文本提示和置信度阈值0.5为准。测试集只有白实线与白虚线真值；其他五类只能观察误检，不能评估正样本IoU。各实验的损失项不同，跨实验应优先比较统一实际IoU和Recall，不直接比较`val/loss`绝对值。

## 当前总体结论

- Base + DETR LoRA仍显著领先，尤其是白虚线召回。
- TinyViT Stage-3比EfficientViT Stage-3道路标线基线更适合当前蒸馏路线。
- 最终输出KD、图像特征KD和扩大图像LoRA覆盖范围均有正收益，但目前都只是渐进提升。
- 当前已完成的最佳轻量方案是TinyViT P2；下一项严格对照是P3图像r16正式训练与统一评测。

## 阅读顺序

1. 先看[Base + DETR LoRA](sam3_detr_exp/README.md)，了解教师模型和效果上限。
2. 再看[EfficientViT Stage-3直接LoRA](sam3_lightweight_stage3_exp/README.md)及其[蒸馏实验](sam3_lightweight_stage3_distill_exp/README.md)。
3. 最后看当前主线[TinyViT Stage-3 P0～P3](sam3_lightweight_tinyvit_stage3_distill_exp/README.md)。

## 环境

当前可运行环境以仓库内`.venv`和根目录[requirements.txt](requirements.txt)为准。具体启动命令、数据格式与模型下载方法请进入相应实验目录查看，避免在根目录复制并维护多份实验细节。
