# P2 图像 LoRA 扩展到 Stage 1/2/3

本实验只改变 TinyViT 图像 LoRA 的覆盖范围：从 stage 2、3 扩展到 stage 1、2、3。LoRA rank 仍为 8，DETR LoRA、损失、缓存、学习率和数据保持不变，用于判断浅层图像特征是否限制车道线能力。

## 权重继承

- 从 `../weights/p1_image_feature_r8.best.pt` 继续训练。
- stage 2、3 的图像 LoRA、DETR LoRA、分类头和分割头直接继承。
- stage 1 的 attention `qkv/proj` 与 MLP `fc1/fc2` 新增 r8 LoRA，A 使用 Kaiming 初始化，B 从零初始化，因此刚开始不会改变 P1 的输出。
- stage 0、卷积、局部卷积、Norm 与 neck 仍然冻结。

这一方案不需要合并或改变已有 r8 的矩阵形状，因此可以安全继承 P1 权重。

## 启动条件

先等待 P1 完成并生成：

```text
sam3_lightweight_tinyvit_stage3_distill_exp/weights/p1_image_feature_r8.best.pt
```

然后执行四卡训练：

```bash
bash sam3_lightweight_tinyvit_stage3_distill_exp/p2_image_stage123/scripts/train_p2_image_stage123.sh
```

日志和权重分别输出到：

```text
../logs/p2_image_stage123/
../weights/p2_image_stage123_r8.best.pt
../weights/p2_image_stage123_r8.pt
```
