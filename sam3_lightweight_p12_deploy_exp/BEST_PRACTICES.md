# SAM3轻量化部署最佳实践

本文是部署的入口：选哪个方案、需要哪些文件、怎么跑、现在能跑多快。
训练与蒸馏请先看[《SAM3微调与轻量蒸馏最佳实践》](../BEST_PRACTICES.md)；
完整精度对照见[PyTorch部署实验](README.md)，ONNX/TensorRT实验见[量化部署记录](QUANTIZATION.md)，
模块级瓶颈定位见[模块耗时分析](profiling.md)。

## 1. 选哪个方案

| 场景 | 方案 | 体积 | 白实线IoU | 白实线Recall | 单提示端到端 |
|---|---|---|---|---|---|
| 7类提示，严格同精度 | PyTorch FP32包 | 238.92 MiB | 0.654836 | 0.714868 | 未在同口径下测 |
| 7类提示，要更小 | PyTorch FP16包＋autocast | 119.60 MiB | 0.654737 | 0.714003 | 63.50 ms |
| 单提示，要最快 | TensorRT FP16引擎 | 1029.84 MiB | 0.648045 | 0.709623 | **24.74 ms** |

精度为白实线单类的10图union-mask值；速度为第5节的GPU张量口径（预处理＋前向＋后处理），
三行可直接对比。FP32包没有同口径实测值——[部署实验README](README.md)第3.2节的110.06ms是
含PIL转张量的另一种口径，两者不能相除。TensorRT引擎体积大于PyTorch包的原因尚未定位，见
[量化部署记录](QUANTIZATION.md)第5节。

三行都满足"能部署"，但约束不同，按需要选：

- **只有PyTorch两条路线支持全部7个道路标线提示**，且部署包由本仓库PyTorch代码构建。
  FP32包在同精度10图、70个提示对照中逐位一致；FP16包有舍入差异，不能称为完全无损。
- **TensorRT引擎把提示写死在图里**，一个引擎只对应一个提示（此处是白实线），换提示必须重新导出；
  精度低于PyTorch基准，且AGX尚未构建和验证。
- **INT8已排除**：两种运行时下精度都不可用，相对FP16也没有延迟收益，见[量化部署记录](QUANTIZATION.md)。

所有数字为A800实测，不能推算AGX帧率。

## 2. 准备文件

从仓库根目录执行。导出需要以下两个源文件：

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

验收、测速和本文示例默认使用历史十图清单
`sam3_lightweight_tinyvit_stage3_distill_exp/tests/output/p12_query_set_best_first10_threshold_05/summary.json`，
每张图的真值由同名`.txt`文件提供，可用`--manifest`替换。

## 3. 路线A：PyTorch 7类部署包

### 3.1 导出与验证

导出器固定7类文本特征、移除运行时MobileCLIP、合并124处LoRA，输出单个文件：

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  sam3_lightweight_p12_deploy_exp/deploy.py export
```

默认输出`weights/p12_fixed_vocab_fp32.pt`。导出器会严格重载并逐张量检查，目标已存在时拒绝覆盖。
随后执行同精度等价验证：

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  sam3_lightweight_p12_deploy_exp/deploy.py verify \
  --warmup 30 --repeats 200
```

需要FP16存储包时执行一次，输出`weights/p12_fixed_vocab_fp16.pt`：

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  sam3_lightweight_p12_deploy_exp/precision_test.py convert
```

FP16文件只是把浮点权重压成半精度存储；当前安全运行方式是恢复成FP32参数并使用FP16 autocast，
不能把文件缩小一半等同于显存或延时缩小一半。

### 3.2 推理

```bash
.venv/bin/python sam3_lightweight_p12_deploy_exp/infer.py \
  --package sam3_lightweight_p12_deploy_exp/weights/p12_fixed_vocab_fp16.pt \
  --precision fp16 \
  --image /path/to/image.jpg \
  --prompt "white solid lane line" \
  --output sam3_lightweight_p12_deploy_exp/tests/output/prediction.png
```

固定词表为白实线、黄实线、白虚线、黄虚线、斑马线、车道护栏和道路齿状标线。未知文本会明确报错；
改词表必须回到带MobileCLIP的源模型重新提取文本特征，不能只修改字符串。推理保存覆盖图和同名NPZ，
颜色只表示预测覆盖，不自动表示正确或误检。同图空几何提示编码会被缓存，加入点、框或mask提示时自动绕过。

## 4. 路线B：TensorRT 单提示引擎

只用于固定白实线单提示、1008×1008输入，需要先有第3.1节导出的FP32部署包。
细节和全部实测见[量化部署记录](QUANTIZATION.md)。

```bash
# 1. 从FP32部署包导出固定形状ONNX。关闭抗锯齿才能兼容TensorRT，会改变mask输出，必须另做精度校验
.venv/bin/python sam3_lightweight_p12_deploy_exp/export_onnx.py \
  --tensorrt-compatible \
  --output sam3_lightweight_p12_deploy_exp/weights/p12_white_solid_trt_fp32.onnx
# 2. 从该ONNX直转FP16
.venv/bin/python sam3_lightweight_p12_deploy_exp/export_fp16_onnx.py \
  --input sam3_lightweight_p12_deploy_exp/weights/p12_white_solid_trt_fp32.onnx \
  --output sam3_lightweight_p12_deploy_exp/weights/p12_white_solid_trt_fp16_direct.onnx
# 3. 构建本机引擎
.venv/bin/python sam3_lightweight_p12_deploy_exp/build_tensorrt.py \
  --onnx sam3_lightweight_p12_deploy_exp/weights/p12_white_solid_trt_fp16_direct.onnx \
  --output sam3_lightweight_p12_deploy_exp/weights/p12_white_solid_trt_fp16_direct.engine
# 4. 单图推理
.venv/bin/python sam3_lightweight_p12_deploy_exp/infer_tensorrt.py \
  --engine sam3_lightweight_p12_deploy_exp/weights/p12_white_solid_trt_fp16_direct.engine \
  --image /path/to/image.png \
  --output sam3_lightweight_p12_deploy_exp/tests/output/prediction.png
```

引擎是A800本机构建物，**AGX需要在目标机重新构建、验证和测速**。构建耗时约250秒。

## 5. 正确测速

不要把模型构建、权重读取与传GPU、磁盘读图或图片解码混入模型推理速度。执行：

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=0 .venv/bin/python \
  sam3_lightweight_p12_deploy_exp/benchmark_latency_stages.py \
  --warmup 20 --repeats 100
```

A800、单提示、3840×2160原图、1008网络输入实测：

| 口径 | PyTorch FP16 autocast | TensorRT FP16引擎 |
|---|---:|---:|
| 预处理：GPU uint8张量缩放与归一化 | 0.78 ms | 0.78 ms |
| 神经网络前向 | 61.63 ms | 22.68 ms |
| ├ 图像网络（TinyViT及P5～P8） | 22.60 ms | 无法拆分 |
| ├ 固定文本查表 | 0.15 ms | 无法拆分 |
| └ Grounding网络 | 38.68 ms | 无法拆分 |
| 后处理：阈值、框变换、mask上采样 | 1.41 ms | 1.28 ms |
| **组合链路实测** | **63.50 ms** | **24.74 ms** |

TensorRT引擎是单一融合图，无法像PyTorch那样把图像网络与Grounding拆开测；TensorRT一列的
命令为`benchmark_engine_stages.py`。分项之和（PyTorch 63.82ms）与组合实测（63.50ms）的差值是
各阶段独立同步的边界开销，比较总耗时请用组合链路值。

PyTorch一列为历史记录值；同会话重测为0.77／61.21／1.41／63.03ms，差异约1%，见
[量化部署记录](QUANTIZATION.md)第4节。

生产视频应尽量让解码器直接输出张量，避免每帧转为PIL：PIL对象组合链路约97.88ms，
但该数字不是纯模型速度。以上均为A800结果，不能推算为AGX实测帧率。

## 6. 已验证的优化与当前边界

- 六个蛇形卷积合计约4.7ms，不是当前首要瓶颈；单提示主要耗时在Grounding网络。
- 同图多提示可缓存固定空几何编码；10图×7提示原始输出严格一致，串行7提示约节省20ms。
- 7提示批量前向由319.05ms降至149.13ms，但输出不是逐位一致，因此仍是可选实验路径。
- 只编译Grounding的单提示试验由63.24ms降至41.21ms，但存在少量实例和mask变化，未接入默认入口。
- 固定白实线的TensorRT单提示引擎已在A800上构建并实测，端到端24.74ms。

**未完成**：AGX实机未构建、未测；TensorRT只覆盖白实线，其余6个提示未验证；7个提示在TensorRT上的
代价未测（7个引擎串行估算约159ms）。当前不能宣称已经达到AGX实时部署。

部署代码、精度结果、模块profile及所有限制统一以[PyTorch部署实验](README.md)、
[量化部署记录](QUANTIZATION.md)和[模块耗时分析](profiling.md)为准。
权重及`tests/output`被Git忽略，不会随源码提交。
