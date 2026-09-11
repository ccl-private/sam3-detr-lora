# SAM3轻量化部署最佳实践

本文面向已经获得轻量模型、希望直接导出和测速的新用户。训练与蒸馏请先看
[《SAM3微调与轻量蒸馏最佳实践》](../BEST_PRACTICES.md)，历史部署消融、完整精度表和实现细节见
[P12固定词表轻量部署实验](README.md)。

## 1. 当前推荐方案

| 项目 | 当前选择 |
|---|---|
| 轻量模型 | TinyViT P12 epoch 19最佳权重 |
| 提示方式 | 固定7类道路标线文本，不输入点、框或mask提示 |
| 默认可靠版本 | 合并LoRA的FP32单文件，238.92 MiB |
| 速度/体积候选 | FP16存储包＋FP16 autocast，119.60 MiB |
| 输入尺寸 | 1008×1008 |
| 置信度阈值 | 0.5 |

FP32部署包在同精度10图、70个提示对照中，分类、Presence、200个候选框、低分辨率mask logits
及最终实例mask全部一致。FP16包平均IoU接近，但存在舍入差异，不能称为完全无损。
当前导出物仍通过本仓库PyTorch代码构建模型，不是独立ONNX或TensorRT引擎。

## 2. 准备文件

从仓库根目录执行。部署导出需要以下两个源文件：

| 文件 | 来源 |
|---|---|
| `sam3_lightweight_stage3_exp/input/efficientsam3_tinyvit_stage3.pt` | [EfficientSAM3官方TinyViT权重](https://huggingface.co/Simon7108528/EfficientSAM3/blob/main/efficientsam3_ft/efficientsam3_tinyvit.pt) |
| `sam3_lightweight_tinyvit_stage3_distill_exp/weights/p12_query_set_distill.best.pt` | 本项目完成P12蒸馏后的epoch 19最佳训练产物，不在Git仓库中 |

官方TinyViT基模可直接下载：

```bash
mkdir -p sam3_lightweight_stage3_exp/input
curl -fL --retry 5 \
  -o sam3_lightweight_stage3_exp/input/efficientsam3_tinyvit_stage3.pt \
  'https://huggingface.co/Simon7108528/EfficientSAM3/resolve/main/efficientsam3_ft/efficientsam3_tinyvit.pt'
sha256sum sam3_lightweight_stage3_exp/input/efficientsam3_tinyvit_stage3.pt
```

预期SHA-256为：

```text
3a52c42f975a9562cb656a57aebb60314d7871c3956e4955c91364a8fe3875dc
```

P12权重不是上游官方模型。若本地没有该文件，需要按照训练最佳实践先完成教师模型和P12蒸馏；
不能只下载官方TinyViT就复现本文效果。

## 3. 导出部署包

导出器会固定7类文本特征、移除运行时MobileCLIP、合并124处LoRA，并把完整推理权重保存为
一个文件：

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  sam3_lightweight_p12_deploy_exp/deploy.py export
```

默认输出：

```text
sam3_lightweight_p12_deploy_exp/weights/p12_fixed_vocab_fp32.pt
```

导出器会严格重载并逐张量检查；若目标文件已经存在会拒绝覆盖。随后执行同精度等价验证：

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  sam3_lightweight_p12_deploy_exp/deploy.py verify \
  --warmup 30 --repeats 200
```

需要FP16存储包时执行一次：

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  sam3_lightweight_p12_deploy_exp/precision_test.py convert
```

默认输出`weights/p12_fixed_vocab_fp16.pt`。FP16文件只是把浮点权重压成半精度存储；当前安全运行
方式会恢复FP32参数并使用FP16 autocast，不能把文件缩小一半等同于显存或延时缩小一半。

## 4. 直接测试单提示

```bash
.venv/bin/python sam3_lightweight_p12_deploy_exp/infer.py \
  --package sam3_lightweight_p12_deploy_exp/weights/p12_fixed_vocab_fp16.pt \
  --precision fp16 \
  --image /path/to/image.jpg \
  --prompt "white solid lane line" \
  --output sam3_lightweight_p12_deploy_exp/tests/output/prediction.png
```

当前固定词表为白实线、黄实线、白虚线、黄虚线、斑马线、车道护栏和道路齿状标线。未知文本会
明确报错；改词表必须回到带MobileCLIP的源模型重新提取文本特征，不能只修改字符串。
推理会保存覆盖图和同名NPZ；颜色只表示预测覆盖，不自动表示正确或误检。

## 5. 正确测速

不要把模型构建、权重读取与传GPU、磁盘读图或图片解码混入模型推理速度。执行：

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=0 .venv/bin/python \
  sam3_lightweight_p12_deploy_exp/benchmark_latency_stages.py \
  --warmup 20 --repeats 100
```

当前A800、FP16、单提示实测：

| 口径 | 平均耗时 |
|---|---:|
| 预处理：GPU uint8张量缩放与归一化 | 0.78 ms |
| 神经网络前向 | 61.63 ms |
| 后处理：阈值、框变换、mask上采样 | 1.41 ms |
| 预处理＋前向＋后处理组合链路 | 63.50 ms |

生产视频应尽量让解码器直接输出张量，避免每帧转为PIL。PIL对象组合链路约97.88ms，但该数字
不是纯模型速度。以上均为A800结果，不能推算为AGX实测帧率。

## 6. 已验证优化与当前边界

- 六个蛇形卷积合计约4.7ms，不是当前首要瓶颈；单提示主要耗时在Grounding网络。
- 同图多提示可缓存固定空几何编码；10图×7提示原始输出严格一致，串行7提示约节省20ms。
- 7提示批量前向由319.05ms降至149.13ms，但输出不是逐位一致，因此仍是可选实验路径。
- 只编译Grounding的单提示试验由63.24ms降至41.21ms，但存在少量实例和mask变化，未接入默认入口。
- TensorRT固定形状引擎和AGX实机测试尚未完成，当前不能宣称已经达到AGX实时部署。

部署代码、精度结果、模块profile及所有限制统一以
[部署实验README](README.md)和[模块耗时分析](profiling.md)为准。权重及`tests/output`被Git忽略，
不会随源码提交。
