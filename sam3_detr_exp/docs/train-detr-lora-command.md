# DETR LoRA 训练命令手册

本文档只说明如何执行 `sam3_detr_exp/train_detr_lora.py`。训练原理和代码结构见
[detr-lora-finetune.md](detr-lora-finetune.md)。

## 1. 启动前检查

在项目根目录执行命令：

```bash
cd /slow_disk/ccl/codes/sam3
```

确认以下文件存在：

- Python 环境：`./.venv/bin/python`
- 训练入口：`sam3_detr_exp/train_detr_lora.py`
- 数据 YAML：由 `--data-yaml` 指定
- SAM3 modular 权重：`sam3_detr_exp/weights_modular/`

查看当前脚本支持的全部参数：

```bash
./.venv/bin/python sam3_detr_exp/train_detr_lora.py --help
```

4 卡训练由 Lightning 自动启动 4 个 DDP worker，不需要使用 `torchrun`。启动日志应出现
4 行 `cuda binding`，且 `local_rank` 和 `current_device` 分别为 0、1、2、3。

## 2. 数据 YAML

训练集、验证集、类别名称和多提示策略均由 YAML 决定。推荐格式：

```yaml
path: /absolute/path/to/dataset
train: train
val: val

names:
  0: first class prompt
  1: second class prompt

prompt_training:
  mode: multi_prompt
  num_negatives: 2
  generic_negatives:
    - person
    - dog
    - chair
```

路径规则：

- `path` 推荐使用绝对路径。
- `train` 和 `val` 相对于 `path` 解析，也可以直接写绝对路径。
- multi-prompt 模式下，每张图片固定使用 YAML 中的全部类别提示。
- 图片中不存在的类别作为数据集内空 target 负提示。
- `num_negatives` 只控制训练阶段额外采样的通用负提示数量。
- 验证阶段不加入随机通用负提示，保证各 epoch 的 `val/loss` 口径固定。

## 3. 常用参数

### 数据与提示词

| 参数 | 含义 | 推荐值 |
|---|---|---|
| `--data-yaml` | 数据配置文件 | 必填，显式指定 |
| `--prompt-mode` | `class_name` 使用 YAML 类别名；`generic` 使用同一个通用提示 | `class_name` |
| `--generic-prompt` | `prompt-mode=generic` 时的文本 | 仅通用提示实验使用 |
| `--max-train-samples` | 最多加载的训练图片数 | 正式训练不设置 |
| `--max-val-samples` | 最多加载的验证图片数 | 正式训练不设置 |

multi-prompt 模式下，`max_*_samples` 和 `batch-size` 都按图片计数，不按展开后的提示词数量计数。

### Loss 与可训练模块

| 参数 | 含义 | 推荐值 |
|---|---|---|
| `--loss-mode sam3` | IABCE/presence、box/GIoU、focal mask、Dice | 正式训练使用 |
| `--loss-mode simple` | 旧版简化 loss | 仅用于旧实验复现 |
| `--train-dot-score` | 同时训练文本与目标查询的打分层 | 检测任务开启 |
| `--train-seg-head` | 同时训练分割头 | 分割任务开启 |
| `--mask-weight` | 简化 loss 的 mask 权重 | `loss-mode=sam3` 时不生效 |

`--decoder-only` 只给 transformer decoder 挂 LoRA；`--attn-only` 排除 FFN。没有明确消融目的时，
两者都不要设置。

### LoRA 与优化器

| 参数 | 含义 | 当前稳定配置 | 高容量对照配置 |
|---|---|---:|---:|
| `--lora-rank` | 低秩维度 | 8 | 16 |
| `--lora-alpha` | LoRA 缩放系数 | 16 | 32 |
| `--lora-dropout` | LoRA dropout | 0.05 | 0.1 |
| `--lr` | AdamW 学习率 | `2e-4` | `1e-4` |
| `--weight-decay` | AdamW weight decay | `1e-2` | `1e-2` |

当前脚本尚未实现 warmup、cosine scheduler 和梯度裁剪。命令中不要填写不存在的参数。

### GPU、batch 与日志

| 参数 | 含义 | 4×A800 推荐值 |
|---|---|---:|
| `--accelerator` | 训练设备类型 | `gpu` |
| `--devices` | DDP GPU 数量 | 4 |
| `--precision` | 混合精度 | `bf16-mixed` |
| `--resolution` | 训练输入边长 | 1008 |
| `--batch-size` | 每卡图片数 | 先用 1 或 2 |
| `--num-workers` | 每个 rank 的 DataLoader worker 数 | 8 |
| `--log-every` | 每隔多少 train step 写一次日志 | 10 |

全局图片 batch 为 `devices × batch-size`。例如 4 卡、每卡 2 张时，全局图片 batch 为 8。
multi-prompt 会继续把每张图展开为多个文本提示，因此不要用单提示训练的显存经验直接设置 batch 8。

### 训练范围与输出

| 参数 | 含义 |
|---|---|
| `--epochs` | 完整遍历训练集的次数 |
| `--dry-run` | 强制只跑 1 个 train batch、1 个 val batch 和 1 个 epoch |
| `--limit-train-batches` | 每个 epoch 使用的训练 batch 比例，正式训练保持 `1.0` |
| `--limit-val-batches` | 每次验证使用的 batch 比例，正式训练保持 `1.0` |
| `--save` | 最后一个 epoch 的 LoRA checkpoint |
| `--best-save` | 最低 `val/loss` 的 checkpoint；不设置时自动在 `--save` 文件名中加入 `.best` |
| `--seed` | 随机种子，默认 42 |

例如 `--save weights/run01.pt` 会默认产生：

- `weights/run01.pt`：最后一个 epoch。
- `weights/run01.best.pt`：验证 loss 最低的 epoch。

推理和正式评估通常应加载 `.best.pt`。每次实验应使用新的输出文件名，否则会覆盖同名旧权重。
当前脚本只保存 LoRA 权重，不保存 optimizer/scheduler 状态，也没有断点续训参数。

Lightning 的逐 step/epoch 指标默认写入 `lightning_logs/version_*/metrics.csv`。如果还需要完整终端日志，
可以在命令末尾使用：

```bash
2>&1 | tee sam3_detr_exp/logs/experiment_name.log
```

运行前需要确保日志目录已经存在。

## 4. 典型示例一：新数据集 4 卡 dry-run

第一次使用新的 YAML 时，先限制样本并执行单步训练。这个命令只验证数据、正负提示、loss、
4 卡 DDP 和 checkpoint 保存，不代表模型效果：

```bash
./.venv/bin/python -u sam3_detr_exp/train_detr_lora.py \
  --data-yaml /absolute/path/to/data.yaml \
  --prompt-mode class_name \
  --loss-mode sam3 \
  --train-dot-score \
  --train-seg-head \
  --accelerator gpu \
  --devices 4 \
  --precision bf16-mixed \
  --resolution 1008 \
  --batch-size 1 \
  --num-workers 0 \
  --max-train-samples 20 \
  --max-val-samples 10 \
  --lora-rank 8 \
  --lora-alpha 16 \
  --lora-dropout 0.05 \
  --save /tmp/detr_lora_dry_run.pt \
  --dry-run
```

通过标准：4 个 rank 均绑定正确 GPU；train/val 各完成一个 batch；loss 有限；没有 OOM、NaN、
unused parameter 或设备不一致报错。

## 5. 典型示例二：4 卡正式训练

下面是当前已经完整训练并验证过的稳定参数。输出名包含实验配置，避免覆盖旧模型：

```bash
./.venv/bin/python -u sam3_detr_exp/train_detr_lora.py \
  --data-yaml sam3_detr_exp/configs/roadline_lora.yaml \
  --prompt-mode class_name \
  --loss-mode sam3 \
  --train-dot-score \
  --train-seg-head \
  --accelerator gpu \
  --devices 4 \
  --precision bf16-mixed \
  --resolution 1008 \
  --batch-size 2 \
  --num-workers 8 \
  --lr 2e-4 \
  --weight-decay 1e-2 \
  --lora-rank 8 \
  --lora-alpha 16 \
  --lora-dropout 0.05 \
  --epochs 20 \
  --log-every 10 \
  --save sam3_detr_exp/weights_lora/roadline_r8_a16_lr2e4.pt
```

如果要做高容量、低学习率对照实验，只替换下面四个参数，并使用新的 `--save` 文件名：

```text
--lora-rank 16
--lora-alpha 32
--lora-dropout 0.1
--lr 1e-4
```

## 6. 常见问题

### 显存不足

优先把 `--batch-size 2` 改为 `--batch-size 1`。不要首先降低 1008 分辨率，因为细车道线的
mask 对分辨率敏感。

### 四张卡显存不同

不同图片的实例数和有效 mask 数量不同，matcher 与分割头的临时显存会变化，因此主要显存不必
完全相同。但启动后应只有 4 个唯一训练 PID，每个 rank 的主要模型显存应位于对应 GPU。

### 终端看不到 loss

训练进度条会原地刷新。需要持久记录时查看 `lightning_logs/version_*/metrics.csv`，或使用 `tee`
保存完整终端输出。`--log-every` 过大时，短 dry-run 不会产生逐 step 日志。

### best loss 波动

当前 best checkpoint 按固定验证提示集合上的 `val/loss` 保存。应按 epoch 比较，不要根据单个
train batch 的 loss 判断是否收敛。
