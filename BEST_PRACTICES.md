# SAM3微调与轻量蒸馏最佳实践

本文面向第一次使用本仓库的开发者，只保留已经完成、可复核且当前效果最好的两条路线：

1. 用SAM3 Base做DETR LoRA微调，追求最高精度；
2. 用上述Base最佳模型作教师，训练TinyViT P12，追求轻量化。

历史实验、失败消融和研究过程仍保留在各实验目录，但不要求新手从P0依次重跑到P12。

## 1. 先看结论

| 目标 | 当前推荐模型 | 固定10图平均IoU | 说明 |
|---|---|---:|---|
| 最高道路标线效果 | Base回溯消融：无域外纯负提示 | **0.7483** | 当前总体最佳，同时保留已测`car`能力 |
| 当前定量最佳轻量模型 | TinyViT P12 | **0.6670** | 比轻量P9提高0.0231，仍比Base低0.0813 |
| 跨场景研究模型 | TinyViT P13-A | 暂无同口径10图结果 | 网图召回增强，但重复Query、类别混淆和置信度失准，不作为默认部署模型 |

这里的10图全部来自同一段无人机视频，只用于历史同口径回归。真正部署前还必须使用自己的独立
测试集评估。若只关心效果，选择Base；若显存、延迟或部署体积更重要，选择P12。

## 2. 环境准备

以下命令均从仓库根目录执行：

```bash
cd /path/to/sam3
python3.13 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

本仓库实测环境是Python 3.13.11、Lightning 2.6.5和CUDA GPU。`requirements.txt`来自实际实验
环境的完整冻结版本；若CUDA、PyTorch或FlashAttention安装失败，应先安装与本机驱动匹配的版本，
再安装其余依赖。正式训练默认使用4卡，单卡可以运行，但需把`--devices`改为1并降低每卡batch。

## 3. 必需模型文件和来源

### 3.1 原始SAM3 Base

- 官方模型页：[facebook/sam3](https://huggingface.co/facebook/sam3)
- 官方文件：[sam3.pt](https://huggingface.co/facebook/sam3/blob/main/sam3.pt)
- 本地目标：仓库根目录`sam3.pt`
- 本地实测文件约3.3 GiB

该仓库可能要求先登录Hugging Face并接受模型许可：

```bash
.venv/bin/hf auth login
.venv/bin/hf download facebook/sam3 sam3.pt --local-dir .
```

如果已经通过其他方式下载，只要最终文件位于`./sam3.pt`即可。不要下载SAM3.1替代这条历史实验
所用的SAM3权重；两者结构和checkpoint不是本实验的同一基线。

### 3.2 官方EfficientSAM3 TinyViT Stage-3

- 作者项目：[SimonZeng7108/efficientsam3](https://github.com/SimonZeng7108/efficientsam3)
- 官方模型库：[Simon7108528/EfficientSAM3](https://huggingface.co/Simon7108528/EfficientSAM3)
- 原始文件：[efficientsam3_tinyvit.pt](https://huggingface.co/Simon7108528/EfficientSAM3/blob/main/efficientsam3_ft/efficientsam3_tinyvit.pt)
- 本地目标：`sam3_lightweight_stage3_exp/input/efficientsam3_tinyvit_stage3.pt`
- SHA-256：`3a52c42f975a9562cb656a57aebb60314d7871c3956e4955c91364a8fe3875dc`
- 远端大小493 MB，即本地约469.98 MiB

直接从原始Hugging Face地址下载：

```bash
mkdir -p sam3_lightweight_stage3_exp/input
curl -fL --retry 5 \
  -o sam3_lightweight_stage3_exp/input/efficientsam3_tinyvit_stage3.pt \
  'https://huggingface.co/Simon7108528/EfficientSAM3/resolve/main/efficientsam3_ft/efficientsam3_tinyvit.pt'
sha256sum sam3_lightweight_stage3_exp/input/efficientsam3_tinyvit_stage3.pt
```

`stage3`只是本项目加入的本地文件名后缀，文件内容与作者的`efficientsam3_tinyvit.pt`相同。

### 3.3 哪些文件不是下载件

以下文件是本项目训练产物，没有包含在Git仓库，也不是上游官方模型：

| 文件 | 如何得到 | 当前本地文件大小 |
|---|---|---:|
| `sam3_detr_exp/weights_modular/*.pt` | 从`sam3.pt`拆分 | 合计约3.3 GiB |
| `sam3_detr_exp/negative_prompt_ablation/weights/roadline_r8_a16_lr2e4_no_generic_negatives.best.pt` | 完成Base最佳实践训练 | 约76 MiB |
| `sam3_lightweight_tinyvit_stage3_distill_exp/weights/p12_query_set_distill.best.pt` | 完成P12蒸馏 | 约140 MiB，含可删除的重复原权重 |
| 教师输出和图像特征缓存 | 运行缓存脚本 | 只用于训练，不进入部署模型 |

因此，新的使用者只下载两个官方基模还不能直接得到本项目最佳结果：必须先训练Base LoRA，随后
才能用它生成教师缓存并训练P12。若以后单独发布本项目训练权重，应在本节补充对应版本、下载链接
和哈希，不能把上游官方模型与本项目微调产物混为一谈。

## 4. 数据格式

训练代码使用YOLO segmentation多边形标注，但目录格式与Ultralytics常见的`images/`、`labels/`
分层不同：当前loader要求图片和同名`.txt`直接放在同一个split目录中。

```text
/data/my_roadline/
  train/
    000001.jpg
    000001.txt
    000002.png
    000002.txt
  val/
    000101.jpg
    000101.txt
```

支持`.jpg`、`.jpeg`、`.png`和`.bmp`。缺少同名`.txt`的图片会被跳过；在`multi_prompt`模式下，
空`.txt`可以作为“所有数据集内类别均不存在”的负样本，但应确认这确实符合标注含义。

每个标签文件一行表示一个实例：

```text
class_id x1 y1 x2 y2 x3 y3 ...
```

- 坐标是相对原图宽高归一化到`[0, 1]`的多边形顶点；
- 至少3个点，即每行至少7列；
- 同一图片、同一类别可以写多行，训练时会组成该文本提示下的多个实例；
- 越界坐标会被裁剪，但不应依赖裁剪修复错误标注；
- 车道线非常细，建议检查多边形在1008×1008缩放后仍有有效宽度。

数据YAML示例：

```yaml
path: /data/my_roadline
train: train
val: val

names:
  0: white solid lane line
  1: yellow solid lane line
  2: white dashed lane line
  3: yellow dashed lane line
  4: zebra crossing
  5: lane barrier
  6: road teeth marking

prompt_training:
  mode: multi_prompt
  num_negatives: 0
  generic_negatives: []
```

最佳实践保留同一数据集内的空目标类别作为内部负提示，但关闭`person/dog/cat...`这类没有正样本
配对的域外纯负提示。历史消融已经验证，长期加入域外纯负提示会破坏相应开放类别能力，且没有改善
道路标线收敛。数据划分应按视频或连续序列分组，不能把相邻帧随机拆到train和val中。

## 5. 路线A：最高效果的Base DETR LoRA

### 5.1 拆分原始模型

Base训练链路读取模块化权重。只需对每份`sam3.pt`执行一次：

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python \
  sam3_detr_exp/run_video_det_modular.py \
  --checkpoint sam3.pt \
  --output-dir sam3_detr_exp/weights_modular
```

该步骤会生成视觉骨干、文本编码器、DETR Encoder/Decoder、分割头、几何编码器、点积分类头和
tracker等10个文件。训练只使用图像检测/分割相关模块，但当前模块化流程会完整导出10个模块。

### 5.2 配置数据

修改：

```text
sam3_detr_exp/negative_prompt_ablation/configs/no_generic_negatives.yaml
```

将`path/train/val/names`替换为自己的数据；保持：

```yaml
prompt_training:
  mode: multi_prompt
  num_negatives: 0
  generic_negatives: []
```

### 5.3 先冒烟，再正式训练

建议先使用少量样本验证数据和DDP链路，命令模板见
[DETR LoRA训练命令手册](sam3_detr_exp/docs/train-detr-lora-command.md#4-典型示例一新数据集-4-卡试运行)。

四卡正式复现实验：

```bash
bash sam3_detr_exp/negative_prompt_ablation/scripts/train_no_generic_negatives.sh
```

核心配置是1008分辨率、4卡、每卡batch 2、bf16、DETR Encoder/Decoder r8 LoRA、`alpha=16`、
学习率`2e-4`，并完整训练点积分类头和分割头，共20轮。输出包括：

```text
sam3_detr_exp/negative_prompt_ablation/weights/
  roadline_r8_a16_lr2e4_no_generic_negatives.pt
  roadline_r8_a16_lr2e4_no_generic_negatives.best.pt
```

`.best.pt`按最低`val/loss`保存。本项目最佳点为epoch 13，而不是最后一轮。

### 5.4 Base单图文本提示推理

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python \
  sam3_detr_exp/run_detr_prompt_inference.py \
  --image /path/to/image.jpg \
  --text "white solid lane line" \
  --threshold 0.5 \
  --lora sam3_detr_exp/negative_prompt_ablation/weights/roadline_r8_a16_lr2e4_no_generic_negatives.best.pt \
  --output /tmp/base_roadline.png
```

若改用任意类别提示，先在独立图片上检查开放类别是否仍正常，不能只看道路标线验证loss。

## 6. 路线B：当前定量最佳轻量模型P12

P12不是从P0一路续训得到。它从作者官方TinyViT Stage-3重新开始，一次挂载P5～P8完整细线结构，
使用路线A得到的新Base最佳模型作为教师，同时加入：

- SAM3真实标签监督；
- 最终输出分类、Presence、框和mask KD；
- 三尺度图像特征KD；
- 全部200个候选的集合/排序KD；
- 7个道路标线提示之间的软关系KD。

### 6.1 同步三份数据路径

将下面三份YAML的`path/train/val/names`设为完全一致：

```text
sam3_detr_exp/negative_prompt_ablation/configs/no_generic_negatives.yaml
sam3_lightweight_tinyvit_stage3_distill_exp/p12_query_set_distill/configs/roadline_no_generic_negatives.yaml
sam3_lightweight_stage3_exp/configs/roadline_lora.yaml
```

第三份配置只被图像特征缓存脚本读取图片路径；P12训练和稠密教师缓存使用第二份配置。正式训练
仍必须保证提示类别顺序一致，并且域外负提示数量为0。

### 6.2 生成两类教师缓存

先生成Base三尺度图像特征缓存：

```bash
bash sam3_lightweight_tinyvit_stage3_distill_exp/p1_image_feature/scripts/cache_teacher_features_4gpu.sh
```

再生成P12的全部200候选教师缓存：

```bash
bash sam3_lightweight_tinyvit_stage3_distill_exp/p12_query_set_distill/scripts/cache_dense_teacher_queries_4gpu.sh
```

脚本支持按图片哈希断点补齐。开始训练前必须确认train和val的缓存文件数与数据图片数一致；缺少
一张就会在训练时抛出`FileNotFoundError`。缓存占用与图片数和分辨率近似线性增长：本项目11694张
图片的图像特征缓存约39 GiB、P12稠密候选缓存约48 GiB，应至少预留100 GiB工作空间。缓存、
教师模型和Base模块化权重都不属于最终轻量推理包。

### 6.3 四卡训练P12

```bash
bash sam3_lightweight_tinyvit_stage3_distill_exp/p12_query_set_distill/scripts/train_p12_query_set_4gpu.sh
```

默认配置为1008分辨率、4卡、每卡batch 4、bf16、20轮，并逐轮保存checkpoint。最终模型位于：

```text
sam3_lightweight_tinyvit_stage3_distill_exp/weights/p12_query_set_distill.best.pt
```

不要按不同实验的`val/loss`绝对值选模型，因为P12比P9多了候选集合和跨提示关系损失。应同时看
`val/supervised`、逐类IoU/Recall和独立跨场景测试。

### 6.4 统一评测

测试目录仍需采用“图片与同名YOLO segmentation标签放在同一目录”的格式。先生成Base对照：

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python \
  sam3_lightweight_stage3_exp/tests/benchmark_base_detr_lora.py \
  --images /path/to/test_dir \
  --lora sam3_detr_exp/negative_prompt_ablation/weights/roadline_r8_a16_lr2e4_no_generic_negatives.best.pt \
  --output /tmp/base_eval \
  --limit 100 \
  --threshold 0.5
```

再评测P12：

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python \
  sam3_lightweight_tinyvit_stage3_distill_exp/tests/benchmark_roadline.py \
  --images /path/to/test_dir \
  --weights sam3_lightweight_tinyvit_stage3_distill_exp/weights/p12_query_set_distill.best.pt \
  --checkpoint sam3_lightweight_stage3_exp/input/efficientsam3_tinyvit_stage3.pt \
  --base-summary /tmp/base_eval/summary.json \
  --output /tmp/p12_eval \
  --limit 100 \
  --threshold 0.5
```

结果包含`summary.json`、`details.csv`和按类别保存的可视化图。绿色为与真值重合，红色为预测但
真值中没有的区域，蓝色为漏检。若测试图没有标签，颜色不能再解释为正确、误检或漏检。

## 7. 部署体积与能力边界

P12沿用P8完整推理结构，新增的候选集合和跨提示关系KD只在训练时使用。当前本地加载方式是
“469.98 MiB官方TinyViT基模 + 约140 MiB训练checkpoint”，其中checkpoint重复保存了约116 MiB
原始权重，不能把两个文件简单相加作为最终模型大小。

当前理论合并体积为：

| 部署形式 | 预计模型张量大小 | 文本能力 |
|---|---:|---|
| P12 FP32 | 约400 MiB | 运行时任意文本 |
| P12 FP16 | 约200 MiB | 运行时任意文本 |
| P12固定词表FP16 | 约119 MiB | 仅预提取的固定提示词 |

正式单文件合并导出器和TensorRT部署尚未完成，上表是按张量组成计算的理论值，不是已经发布的
部署文件。P5～P8细线分支无法像LoRA一样代数折叠，但可作为约5.27 MiB结构权重放入同一推理包。
完整说明见[模型体积与合并分析](sam3_lightweight_tinyvit_stage3_distill_exp/模型体积与合并分析.md)。

## 8. 新手最容易踩的坑

1. 不要加入没有正样本配对的域外纯负提示；内部空类别提示仍应保留。
2. 不要把图片和标签放在独立`images/`、`labels/`目录，当前loader不会这样查找。
3. 不要随机打散视频相邻帧后再划分train/val，应按视频或连续片段隔离。
4. 不要把P12的总`val/loss`与P9或Base直接比较，损失项不同。
5. 不要把P13-A当作已经优于P12的部署模型；它目前是“召回增强、校准失败”的研究分支。
6. 不要用最后一轮替代最佳轮；Base最佳是epoch 13，P12当前最佳是epoch 19。
7. 不要把教师缓存、逐轮checkpoint和项目目录占用算入最终推理模型大小。
8. 先在自己的独立测试集上扫描阈值，再固定部署阈值；历史0.5只用于统一对比。

## 9. 进一步阅读

- [SAM3 Base模块化与DETR LoRA](sam3_detr_exp/README.md)
- [Base无域外负提示最佳实验](sam3_detr_exp/negative_prompt_ablation/README.md)
- [DETR LoRA训练命令手册](sam3_detr_exp/docs/train-detr-lora-command.md)
- [TinyViT Stage-3实验总览](sam3_lightweight_tinyvit_stage3_distill_exp/README.md)
- [P12候选集合与跨提示关系蒸馏](sam3_lightweight_tinyvit_stage3_distill_exp/p12_query_set_distill/README.md)
- [P13-A无标签蒸馏及校准问题](sam3_lightweight_tinyvit_stage3_distill_exp/p13_unlabeled_output_distill/README.md)
