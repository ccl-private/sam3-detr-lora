# SAM3道路标线轻量化实验总览

本仓库的目标是把SAM3 Base在道路标线文本提示分割上的能力迁移到轻量模型。实验不是五个互不相关的目录，而是一条按结果逐步调整的路线：

```text
0. Base + DETR LoRA：建立效果上限和蒸馏教师
   ↓
1. 早期TinyViT-S：验证轻量模型、固定词表和LoRA训练链路
   ↓
2. EfficientViT Stage-3直接LoRA：使用作者更新的官方轻量模型完整训练
   ↓ 效果仍明显低于Base
3. EfficientViT Stage-3 P0蒸馏：尝试用Base最终输出监督学生
   ↓ 验证loss未持续改善，当前方案又没有训练图像编码器
4. 切换到TinyViT Stage-3蒸馏主线
   P0 最终输出KD
    ↓
   P1 增加三尺度图像特征KD
    ↓
   P2 图像LoRA从Stage 2/3扩展到Stage 1/2/3
    ↓
   P3 图像LoRA从r8提高到r16（待正式训练）
```

下面按真实实验先后说明每一步为什么开始、从哪里继续以及为什么进入下一步。详细命令、代码结构和完整指标放在对应实验目录的README中。

## 0. 建立Base教师和效果上限

对应目录：[sam3_detr_exp](sam3_detr_exp/README.md)

- 起点：原始SAM3 Base模块化权重。
- 改动：在DETR Encoder/Decoder挂载r8 LoRA，同时训练点积分类头和分割头。
- 训练：日志显示完成8轮验证（epoch 0～7），epoch 8未完成；最佳验证loss出现在epoch 4。现有记录没有注明提前停止原因。
- 统一10图结果：白实线IoU/Recall为0.7235/0.7883，白虚线为0.6808/0.8226，两类平均IoU为0.7021。
- 结论：这是当前道路标线效果上限，后续蒸馏都使用该最佳LoRA作为教师。

## 1. 早期TinyViT-S可行性实验

对应目录：[sam3_lightweight_exp](sam3_lightweight_exp/README.md)

- 起点：早期`efficient_sam3_tinyvit_s.pt`。
- 改动：将大文本编码器替换为预提取固定词表；训练DETR r8 LoRA、点积分类头和分割头。
- 训练：原计划3轮，只完成epoch 0便主动停止；验证loss为9.3882。
- 测试：没有计算统一IoU/Recall，只记录阈值0.5下白实线77个、白虚线49个检测，平均置信度分别为0.5956和0.6141。
- 结论：证明了固定提示轻量部署与LoRA训练链路可行，但单轮结果不能参与正式效果排名。
- 为什么继续下一步：作者随后发布了更完整的Stage-3轻量模型，因此不再围绕这份早期权重继续训练。

## 2. EfficientViT Stage-3直接LoRA

对应目录：[sam3_lightweight_stage3_exp](sam3_lightweight_stage3_exp/README.md)

- 起点：作者官方Stage-3 EV-M，即EfficientViT-B1图像编码器和MobileCLIP-S0文本编码器。
- 改动：冻结图像和文本编码器；训练DETR r8 LoRA、点积分类头和分割头。
- 训练：完整完成20轮（epoch 0～19），没有提前停止；最佳验证loss为epoch 10的7.1777。
- 统一10图结果：白实线IoU/Recall为0.3905/0.5137，白虚线为0.2153/0.2403，两类平均IoU为0.3029。
- 结论：完整训练后仍明显低于Base，白虚线召回差距最大。
- 为什么继续下一步：直接LoRA不足，因此尝试用Base教师提供分类、presence、框和mask软目标。

## 3. EfficientViT Stage-3 P0输出蒸馏

对应目录：[sam3_lightweight_stage3_distill_exp](sam3_lightweight_stage3_distill_exp/README.md)

- 学生起点：第2步的EfficientViT Stage-3最佳LoRA。
- 教师：第0步的Base + DETR最佳LoRA。
- 改动：保留真实标签监督，增加最终层分类、presence、框和低分辨率mask KD；学生仍只训练DETR r8 LoRA和两个输出头。
- 训练：计划10轮，完成8轮验证（epoch 0～7），epoch 8未完成后主动停止；最低验证loss为epoch 1的11.1575，之后没有持续改善。
- 测试：当前目录没有保留可复核的统一10图IoU/Recall，因此不能量化相对第2步的实际收益。
- 结论：输出蒸馏链路能够训练，但现有证据不能证明它优于未蒸馏的EfficientViT Stage-3。
- 为什么转向TinyViT：这次EfficientViT方案冻结图像编码器、只调整DETR，无法充分弥补视觉表征差距；EfficientViT又包含较多卷积结构，不能直接照搬面向attention/MLP线性层的图像LoRA方案，因此改用官方TinyViT Stage-3继续验证图像侧LoRA。

## 4. TinyViT Stage-3连续蒸馏主线

总目录：[sam3_lightweight_tinyvit_stage3_distill_exp](sam3_lightweight_tinyvit_stage3_distill_exp/README.md)

这一阶段不是四个独立实验，而是连续续训关系：

```text
官方TinyViT Stage-3 + 兼容的Stage-3 DETR/输出头
  → P0最佳
  → P1从P0最佳继续
  → P2从P1最佳继续
  → P3从P2最佳转换并继续
```

### 4.1 P0：最终输出KD与图像LoRA

- 位置：[实验根目录](sam3_lightweight_tinyvit_stage3_distill_exp/README.md)
- 起点：官方TinyViT Stage-3图像骨干；兼容的DETR和输出头继承Stage-3学生LoRA；教师仍为Base最佳LoRA。
- 改动：TinyViT Stage 2/3图像r8 LoRA、DETR r8 LoRA、两个输出头，加最终输出KD。
- 训练：计划10轮，完成7轮（epoch 0～6）后主动停止；最佳点为epoch 6，验证loss为10.0117。
- 结果：白实线IoU/Recall为0.5187/0.6676，白虚线为0.3400/0.4139，平均IoU为0.4293。
- 结论：明显超过EfficientViT Stage-3，但仍远低于Base，因此下一步直接蒸馏图像特征。

### 4.2 P1：增加三尺度图像特征KD

- 位置：[p1_image_feature](sam3_lightweight_tinyvit_stage3_distill_exp/p1_image_feature/README.md)
- 起点：P0最佳权重。
- 改动：保留P0全部训练项，增加三尺度FPN图像特征KD；图像LoRA仍只覆盖Stage 2/3。
- 训练：计划10轮，完成7轮（epoch 0～6）后主动停止；最佳点为epoch 6，验证loss为10.2271。
- 结果：白实线IoU/Recall为0.5343/0.6863，白虚线为0.3490/0.4279，平均IoU为0.4417。
- 结论：相对P0平均IoU提高0.0124，证明图像特征蒸馏有效，但提升较小，因此下一步扩大图像LoRA覆盖范围。

### 4.3 P2：图像LoRA扩展到Stage 1/2/3

- 位置：[p2_image_stage123](sam3_lightweight_tinyvit_stage3_distill_exp/p2_image_stage123/README.md)
- 起点：P1最佳权重。
- 改动：保持蒸馏损失不变，把图像r8 LoRA从Stage 2/3扩展到Stage 1/2/3。
- 训练：计划10轮，完成6轮（epoch 0～5）后主动停止；最佳点为epoch 5，验证loss为10.1169。
- 结果：白实线IoU/Recall为0.5454/0.7145，白虚线为0.3652/0.4461，平均IoU为0.4553。
- 结论：相对P1平均IoU提高0.0136，是当前最佳轻量结果，但仍没有质的飞跃。由于P2从P1继续训练，收益不能完全归因于新增Stage 1 LoRA。

### 4.4 P3：图像LoRA从r8提高到r16

- 位置：[p3_image_r16](sam3_lightweight_tinyvit_stage3_distill_exp/p3_image_r16/README.md)
- 起点：将P2最佳权重中的图像r8增量合并进基础权重，再挂载零增量图像r16；DETR r8和输出头继续继承P2。
- 改动：只提高图像LoRA秩，蒸馏损失和DETR配置不变。
- 训练：正式训练尚未开始，只完成1个训练step和1个验证step的冒烟测试。
- 测试：转换前后单图白实线IoU为0.5373→0.5375，白虚线为0.3978→0.4012，属于数值与阈值边界波动。
- 结论：目前只能证明权重转换和训练链路正确，不能判断r16是否有效。

## 统一结果对比

统一结果使用相同10张道路图片、相同文本提示和置信度阈值0.5。测试集只有白实线和白虚线真值，其他类别只能观察误检。

| 阶段 | 所属目录 | 白实线IoU | 白实线Recall | 白虚线IoU | 白虚线Recall | 平均IoU |
|---|---|---:|---:|---:|---:|---:|
| Base + DETR LoRA | [sam3_detr_exp](sam3_detr_exp/README.md) | 0.7235 | 0.7883 | 0.6808 | 0.8226 | 0.7021 |
| EfficientViT Stage-3 LoRA | [sam3_lightweight_stage3_exp](sam3_lightweight_stage3_exp/README.md) | 0.3905 | 0.5137 | 0.2153 | 0.2403 | 0.3029 |
| TinyViT P0 | [TinyViT实验根目录](sam3_lightweight_tinyvit_stage3_distill_exp/README.md) | 0.5187 | 0.6676 | 0.3400 | 0.4139 | 0.4293 |
| TinyViT P1 | [p1_image_feature](sam3_lightweight_tinyvit_stage3_distill_exp/p1_image_feature/README.md) | 0.5343 | 0.6863 | 0.3490 | 0.4279 | 0.4417 |
| TinyViT P2 | [p2_image_stage123](sam3_lightweight_tinyvit_stage3_distill_exp/p2_image_stage123/README.md) | 0.5454 | 0.7145 | 0.3652 | 0.4461 | 0.4553 |

早期TinyViT-S和EfficientViT P0蒸馏没有可复核的统一IoU/Recall，因此不放入数值排名。不同阶段的loss组成不同，跨实验应比较上表的实际IoU和Recall，不能直接比较`val/loss`。

## 五个实验目录的职责

| 目录 | 在路线中的位置 |
|---|---|
| [sam3_detr_exp](sam3_detr_exp/README.md) | 第0步：Base教师和效果上限 |
| [sam3_lightweight_exp](sam3_lightweight_exp/README.md) | 第1步：早期轻量化可行性验证 |
| [sam3_lightweight_stage3_exp](sam3_lightweight_stage3_exp/README.md) | 第2步：EfficientViT Stage-3直接LoRA基线 |
| [sam3_lightweight_stage3_distill_exp](sam3_lightweight_stage3_distill_exp/README.md) | 第3步：EfficientViT最终输出蒸馏 |
| [sam3_lightweight_tinyvit_stage3_distill_exp](sam3_lightweight_tinyvit_stage3_distill_exp/README.md) | 第4步：TinyViT P0～P3连续蒸馏主线 |

## 当前结论与下一步

- Base + DETR LoRA仍是效果上限，主要差距集中在白虚线召回。
- EfficientViT路线完整训练后效果不足，输出蒸馏也没有形成可量化改善，因此已经转向TinyViT。
- TinyViT的最终输出KD、图像特征KD和扩大图像LoRA覆盖范围都带来连续小幅提升，但尚未接近Base。
- 当前已完成的最佳轻量方案是P2；下一步是正式训练P3，并按同一10图口径判断图像r16是否带来超过P2续训噪声的增益。

## 环境与目录约束

当前运行环境以仓库内`.venv`和根目录[requirements.txt](requirements.txt)为准。根README只记录路线和横向结果；模型结构、下载、训练命令、日志与完整测试说明放在对应实验目录。缓存、权重、日志和`tests/output`不提交Git。
