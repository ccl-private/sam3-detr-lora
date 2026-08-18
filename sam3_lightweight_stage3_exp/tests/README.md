# Stage-3 测试目录

本目录只保存训练无关的测试代码、测试输入、评测文档、指标和可视化结果。LoRA 训练代码、训练配置、训练日志和训练权重均保留在上一级实验目录。

## 人物提示测试

从 `/slow_disk/ccl/codes/sam3` 执行：

```bash
bash sam3_lightweight_stage3_exp/tests/run_person.sh
```

默认输入为 `tests/input/sample_person.png`，结果写入 `tests/output/`。

## 道路标线对比测试

```bash
CUDA_VISIBLE_DEVICES=0 /slow_disk/ccl/codes/efficientsam3/.venv/bin/python \
  sam3_lightweight_stage3_exp/tests/benchmark_roadline.py \
  --images /mnt/mnt108_hdd/biaozhu/labeled/shenxing/12/segment/roadline/20260106
```

默认输出目录为 `tests/output/roadline_comparison/`。完整测试结论见 [ROADLINE_RESULTS.md](ROADLINE_RESULTS.md)。

可视化中只有绿色表示预测正确；红色表示误检，蓝色表示漏检。
