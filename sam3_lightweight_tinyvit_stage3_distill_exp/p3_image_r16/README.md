# P3 图像LoRA r16实验

本实验从P2最佳权重继续，只提高图像侧LoRA容量：

- TinyViT stage 1、2、3：r8、alpha16改为r16、alpha32。
- DETR LoRA继续保持r8、alpha16。
- 三尺度图像特征蒸馏和P0最终输出蒸馏保持不变。

由于r8与r16矩阵形状不同，不能直接加载。必须先运行转换脚本，把P2图像r8增量合并进基础权重，移除旧图像LoRA，再挂载零增量初始化的r16 LoRA。DETR r8与训练头不做转换，直接继承。

## 执行顺序

```bash
bash sam3_lightweight_tinyvit_stage3_distill_exp/p3_image_r16/scripts/convert_image_lora_rank.sh
bash sam3_lightweight_tinyvit_stage3_distill_exp/p3_image_r16/scripts/train_p3_image_r16.sh
```

输出：

```text
../weights/p3_image_r16_init.pt
../weights/p3_image_r16.best.pt
../weights/p3_image_r16.pt
../logs/p3_image_r16/
```
