# SAM3 Lightweight Stage-3 蒸馏实验

本目录用于把已完成道路标线 LoRA 微调的 SAM3 Base DETR 能力蒸馏到 EfficientSAM3 Stage-3 EV-M。

## 目录边界

所有蒸馏相关内容必须放在本目录，包括训练和验证代码、配置、教师缓存、特征适配器、损失、日志、权重、测试及文档。不要把蒸馏代码写回原实验目录。

- `../sam3_detr_exp/`：Base DETR 教师及已有 LoRA
- `../sam3_lightweight_stage3_exp/`：未蒸馏 Stage-3 基线
- 本目录：仅负责蒸馏

P0 最终层 LoRA 蒸馏已经实现。详细设计见 [蒸馏方案讨论.md](./蒸馏方案讨论.md)。

## P0 实现

- `cache_teacher.py`：使用 Base DETR 最佳 LoRA 生成离线教师缓存
- `train_p0.py`：读取缓存训练 Stage-3 r8 LoRA
- `configs/p0_r8.yaml`：固定实验参数和损失权重
- `scripts/cache_teacher.sh`：生成完整训练集与验证集缓存
- `scripts/train_p0.sh`：使用 4 张 GPU 运行 10 轮 P0 训练

P0 蒸馏最终层的 Query 分类、presence、框和低分辨率 mask；真实标签侧继续使用完整 SAM3 O2O、aux 与 O2M 监督。教师与学生通过相同真实实例索引对齐，不按 Query 顺序直接对应。

四卡正式训练默认每卡 batch size 为 4，全局 batch size 为 16。单卡冒烟已验证 batch size 2、4 和 8 均可完成前向、反向与验证；正式值选择 4，为实例数较多的样本保留显存余量。

运行顺序：

```bash
bash sam3_lightweight_stage3_distill_exp/scripts/cache_teacher.sh
bash sam3_lightweight_stage3_distill_exp/scripts/train_p0.sh
```

教师缓存、日志和训练权重均被本目录 `.gitignore` 排除。

当前数据包含 9351 张训练图和 2343 张验证图，完整缓存属于长任务。教师缓存保留配置中的全部类别提示和全部通用负提示，不减少提示词。`cache_teacher.py` 使用 `--image-batch-size` 批量编码图片、`--prompt-batch-size` 批量解码提示，并通过多进程 DataLoader 并行生成真值 mask。它还支持通过 `--num-shards` 与 `--shard-index` 让不同 GPU 使用相同 `--cache-root` 并行写入互不重叠的缓存文件。例如四卡分别运行分片 0～3：

```bash
CUDA_VISIBLE_DEVICES=0 ./.venv/bin/python -u sam3_lightweight_stage3_distill_exp/cache_teacher.py ... --num-shards 4 --shard-index 0
CUDA_VISIBLE_DEVICES=1 ./.venv/bin/python -u sam3_lightweight_stage3_distill_exp/cache_teacher.py ... --num-shards 4 --shard-index 1
CUDA_VISIBLE_DEVICES=2 ./.venv/bin/python -u sam3_lightweight_stage3_distill_exp/cache_teacher.py ... --num-shards 4 --shard-index 2
CUDA_VISIBLE_DEVICES=3 ./.venv/bin/python -u sam3_lightweight_stage3_distill_exp/cache_teacher.py ... --num-shards 4 --shard-index 3
```

省略号处使用 `scripts/cache_teacher.sh` 中的相同数据、教师权重与缓存参数。

## P0实际训练进度与结论

- 学生起点：`../sam3_lightweight_stage3_exp/weights_lora/roadline_stage3_ev_m.best.pt`。
- 教师：`../sam3_detr_exp/weights_lora/roadline_r8_a16_lr2e4.best.pt`。
- 计划轮数：10轮。
- 实际进度：`logs/p0/lightning_logs/version_5/metrics.csv`完成epoch 0～7共8轮验证；epoch 8未完成验证后训练被主动停止。
- 最低验证损失：epoch 1的11.1575；此后没有形成持续下降趋势。
- 统一实测：当前目录没有保留可复核的固定10图IoU、Precision和Recall结果。

阶段结论：P0已经证明最终输出蒸馏链路可以完整训练，但现有验证损失没有表现出持续改善，同时缺少统一10图量化结果。因此目前不能严谨声称它优于未蒸馏Stage-3 LoRA，只能记录为“训练完成8轮后因未见明显改善而停止”。后续若恢复该路线，必须用相同10张图、相同提示和阈值补测后再下效果结论。

## 计划结构

```text
sam3_lightweight_stage3_distill_exp/
├── README.md
├── 蒸馏方案讨论.md
├── configs/                 # 配置
├── scripts/                 # 训练、缓存、导出
├── models/                  # 教师、学生及适配器
├── losses/                  # 蒸馏损失
├── tests/                   # 测试代码
│   └── output/              # 不提交 Git
├── cache/                   # 教师缓存，不提交 Git
├── logs/                    # 不提交 Git
└── weights/                 # 不提交 Git
```
