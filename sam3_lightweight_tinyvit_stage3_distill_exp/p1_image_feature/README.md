# P1 图像编码器多尺度特征蒸馏

本目录只实现 TinyViT Stage-3 的图像特征蒸馏。它从 P0 最佳权重继续训练，并保留 P0 最终输出蒸馏；本阶段暂不加入 DETR Decoder 逐层蒸馏。

## 对齐位置

Base 与 TinyViT 经过各自图像骨干和 neck 后，都输出三层同形 FPN 特征：

```text
256×288×288
256×144×144
256×72×72
```

因此第一版不增加投影适配器，直接对齐归一化后的同通道特征。为控制缓存大小，教师特征统一平均池化 4 倍后以 FP16 保存，对应 `72×72、36×36、18×18`。全量训练集和验证集预计占用约 41 GB，而原尺寸缓存约 650 GB。

## 损失

每层使用逐像素通道归一化 cosine loss，三层等权平均。真实标注 mask 覆盖区域默认权重为背景的 4 倍：

```text
L_total = L_supervised + λ_kd × (L_P0 + λ_image × L_image_feature)
```

默认 `λ_image=1.0`，沿用 P0 的 KD warmup。TinyViT 仍只训练 stage 2、3 的图像 LoRA、DETR LoRA、点积分类头和分割头。

## 运行

先缓存教师图像特征：

```bash
bash sam3_lightweight_tinyvit_stage3_distill_exp/p1_image_feature/scripts/cache_teacher_features.sh
```

缓存可以使用 `--num-shards 4 --shard-index 0..3` 分到四张 GPU 并行生成，所有分片写入同一个缓存目录。

已经提供四卡分片入口：

```bash
bash sam3_lightweight_tinyvit_stage3_distill_exp/p1_image_feature/scripts/cache_teacher_features_4gpu.sh
```

缓存完成后四卡训练：

```bash
bash sam3_lightweight_tinyvit_stage3_distill_exp/p1_image_feature/scripts/train_p1_image_feature.sh
```

缓存、日志和权重分别写入实验根目录的 `cache/p1_image_features`、`logs/p1_image_feature` 和 `weights/p1_image_feature_r8*`，均不提交 Git。
