# P12固定词表轻量部署实验

只想按当前推荐路径导出和测速时，请先看[轻量化部署最佳实践](BEST_PRACTICES.md)；本文继续保留
完整实验过程、精度对照和历史口径。

分模块性能定位见[模块耗时分析](profiling.md)：当前A800上的蛇形卷积不是主要端到端瓶颈。

本目录与训练实验同级，所有部署新增代码、导出权重和测试记录只放这里。只读复用相邻实验的
模型构建及P5～P8结构代码。导出包不需要原始基模或LoRA文件，但运行仍需要本项目及相邻
EfficientSAM3源码和依赖；当前还不是独立TensorRT引擎。

## 基线与阶段

基线为官方TinyViT Stage-3加P12 epoch 19最佳权重，固定7个道路标线提示，1008分辨率、阈值0.5。
文本使用P12自身MobileCLIP-S0逐词提取的特征。DETR、分割头、P5～P8结构和几何编码器保留。

| 阶段 | 内容 | 状态 |
|---|---|---|
| E0～E2 | 原始对照、固定文本、合并124处LoRA、FP32单文件 | 已实现，实际238.92 MiB，完整评测见下文 |
| E3 | 同一图片缓存固定空几何提示编码 | 已实现；保留编码器，7提示约节省20ms，原始输出严格一致 |
| E4 | FP16存储、FP16混合精度与历史BF16计算对照 | 已完成；实际119.60 MiB，平均IoU接近但并非逐位无损 |
| E5 | 预处理、网络前向、后处理及组合链路分项测速 | 已实现，CUDA同步计时 |
| E6 | 7提示PyTorch批处理、ONNX/TensorRT、AGX实测 | PyTorch批处理已评测；导出和AGX仍待执行 |

## 执行

从仓库根目录执行，默认使用CUDA 0。权重及测试产物被Git忽略。

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python sam3_lightweight_p12_deploy_exp/deploy.py export
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python sam3_lightweight_p12_deploy_exp/deploy.py verify --warmup 30 --repeats 200
```

导出文件为`weights/p12_fixed_vocab_fp32.pt`。导出器拒绝覆盖已存在文件；重新导出时使用
`--package`指定新名称。包中包含完整推理权重、固定文本表、结构配置及源模型SHA-256。
加载时严格检查所有参数名及形状，并已检查序列化前后每个张量完全一致。

默认验收图片清单取自历史P12十图报告，使用`--manifest`可指定自己的JSON：

```json
{"images": ["/absolute/path/to/image1.png", "/absolute/path/to/image2.png"]}
```

对照包含全部200候选的分类、Presence、归一化框和低分辨率mask logits，以及阈值后的实例mask。
当前严格测试使用FP32、关闭TF32；历史P12的0.6670来自BF16混合精度，因此两者数值口径应分别记录。
如果相同精度下逐实例mask完全一致，则同一标注上的IoU也完全一致，但不据此假定跨精度等价。

历史整链路测速每次重新处理已解码的PIL对象；7提示复用图像特征、串行解码，包含PIL转张量、
缩放归一化和mask后处理，不包含模型构建、权重读取/传GPU、磁盘读图、图片解码及文件保存。
因此历史表不能称为纯模型推理速度。正式分项口径见下表及`profiling.md`。
当前两个对照模型同时驻留GPU，显存峰值不能作为单模型部署显存报告。

## 单图使用

```bash
.venv/bin/python sam3_lightweight_p12_deploy_exp/infer.py \
  --image /path/to/image.jpg \
  --prompt "white solid lane line" \
  --output sam3_lightweight_p12_deploy_exp/tests/output/prediction.png
```

输出预测覆盖图和同名NPZ，包含mask、框、分数。覆盖颜色不表示正确或误检。
支持白/黄实线、白/黄虚线、斑马线、护栏、道路齿状标线；具体文本列表见`deploy.py`中的
`PROMPTS`。未知词明确报错。切换词表需从原模型重新导出，不能改名后复用原特征。
部署推理入口已启用同图空几何编码缓存；若后续加入点、框或mask提示会自动绕过缓存。

## 当前结果

### 正式单提示延时口径

A800、FP16 autocast、固定白实线提示、1008网络输入，预热20次、重复100次。原始图片为
3840×2160，文件读取和解码已在计时前完成：

| 阶段 | 平均耗时 | 边界 |
|---|---:|---|
| 预处理（上游GPU uint8张量） | 0.78ms | 缩放、归一化、增加batch维 |
| 图像网络 | 22.60ms | TinyViT及P5/P6/P7/P8分支 |
| 固定文本查表 | 0.15ms | MobileCLIP已移除，不是文本编码 |
| Grounding网络 | 38.68ms | 几何编码、Transformer encoder/decoder、分类/框/分割头 |
| **神经网络前向合计** | **61.63ms** | 图像网络＋文本查表＋Grounding，输出低分辨率预测 |
| 后处理 | 1.41ms | 阈值、框变换、mask上采样到原图 |
| **GPU张量组合链路** | **63.50ms** | 预处理＋网络＋后处理的独立组合实测 |
| PIL对象组合链路 | 97.88ms | 额外包含PIL转张量，不含读文件和解码 |

各阶段单独同步会引入边界开销，分项之和不替代组合链路实测。比较模型速度时优先报告
61.63ms；要求最终原图mask时同时报告后处理及63.50ms组合链路。脚本为
`benchmark_latency_stages.py`，结果为`tests/output/latency_stages/report.json`。

空几何缓存的单提示对照96.40→96.54ms只是0.14ms波动：首个提示仍需完整计算几何编码，
没有可复用项，同时多一次缓存判断。它不是模型或精度下降；缓存只对同图第二个及后续提示生效。

2026-09-10完成E0～E2导出及正式测试。单文件238.9243 MiB，实际参数62,374,229；合并124处LoRA。
完整十图、70组提示中，分类logits、Presence logits、全部候选框、低分辨率mask logits的最大
绝对误差均为0，最终逐实例mask全部完全一致，检测数量一致。因此这批图同精度下的IoU不变。
这不是跨FP32/BF16/FP16比较，也不代表所有未来输入都有浮点逐位一致保证。

NVIDIA A800 80GB PCIe，FP32关闭TF32，batch 1，预热30次、重复200次：

| 版本与提示数 | 平均毫秒 | P50毫秒 | P95毫秒 | 相对原版耗时减少 |
|---|---:|---:|---:|---:|
| 原P12，1提示 | 131.70 | 131.47 | 132.49 | — |
| 固定文本并合并，1提示 | 124.83 | 124.70 | 125.61 | 5.22% |
| 原P12，7提示串行 | 516.95 | 514.42 | 519.37 | — |
| 固定文本并合并，7提示串行 | 470.37 | 470.00 | 472.98 | 9.01% |

单提示约8.01张/秒，7提示约2.13张/秒。这是历史FP32处理器链路速度，不是纯网络或AGX速度。
压缩后的文件显著变小，但文本查表和LoRA合并只有有限端到端加速；后续重点是FP16与解码/视觉
计算优化。暂不宣称实时部署已经完成。

完整十图输出及正式测速保存于`tests/output/fp32_report.json`，单图入口也已通过实测，输出为
`tests/output/single_image.png`和同名NPZ。代码与本文可提交Git，权重和测试产物不提交。
没有重新训练、裁剪视觉层或更换蛇形卷积；文件压缩主要来自删除MobileCLIP、重复原权重和tracker备份。

## FP16存储与计算实测

2026-09-10新增`weights/p12_fixed_vocab_fp16.pt`，实测119.5983 MiB。从通过等价验证的FP32
包转换，浮点张量舍入到FP16，整型/布尔张量保持不变。重载时逐张量验证与舍入后权重一致。

注意：FP16文件大小、运行参数精度、算子计算精度是三件事。当前安全运行方式是将权重恢复成FP32，
再使用`torch.autocast(..., dtype=torch.float16)`选择算子精度。DSConv网格坐标和部分归一化仍用
FP32。因此文件缩小一半不代表运行显存也缩小一半；没有直接对整模型执行`model.half()`。

本轮四组均使用同一10图、阈值0.5、关闭TF32、4个CPU线程、每张GPU独立运行一个模型。
速度在首张4K图片上预热30次、重复200次，包含从已解码PIL对象开始的预处理和原图尺寸
mask后处理，不包含模型/权重加载、磁盘读图与图片解码；它是同轮精度模式对照，不是纯网络延时。

| 版本 | 文件MiB | 白实线IoU | 白虚线IoU | 平均IoU | 白实线Recall | 白虚线Recall | 单提示ms | 7提示ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 原P12，历史BF16计算方式重测 | 双文件 | 0.657648 | 0.676723 | 0.667186 | 0.718439 | 0.775614 | 97.99 | 397.02 |
| 固定词表FP32包，FP32计算 | 238.92 | 0.654836 | 0.678788 | 0.666812 | 0.714868 | 0.777684 | 110.06 | 451.41 |
| FP16包，恢复FP32计算 | 119.60 | 0.655350 | 0.678889 | 0.667120 | 0.714774 | 0.777952 | 109.62 | 449.08 |
| FP16包，FP16混合精度计算 | 119.60 | 0.654737 | 0.678783 | 0.666760 | 0.714003 | 0.777930 | 80.28 | 328.61 |

原P12本轮0.667186与历史记录0.666986存在约0.0002差异；本轮关闭TF32，计算环境及执行设置
不与历史记录逐项锁定，因此精度差异统一以本轮重测值为对照，不覆盖历史结果。
本轮每卡单模型，和上一节两个模型同卡驻留的计时环境不同，速度比例使用本轮内部比较。

FP16混合精度相对本轮原P12 BF16：单提示耗时减少18.08%，7提示减少17.23%，约12.46张/秒
和3.04张/秒；平均IoU下降0.000426。相对FP32固定词表基准，平均IoU只下降0.000053，
白实线Recall下降0.000866。

| 版本 | 单提示P50/P95 ms | 7提示P50/P95 ms | 峰值分配显存MiB |
|---|---|---|---:|
| 原P12 BF16 | 97.82 / 99.41 | 397.03 / 400.27 | 3712.25 |
| 固定FP32 | 109.77 / 111.88 | 451.10 / 453.25 | 3690.57 |
| FP16存储、FP32计算 | 109.42 / 110.67 | 448.76 / 451.01 | 3690.57 |
| FP16混合精度 | 80.06 / 81.38 | 325.42 / 342.31 | 3623.81 |

峰值使用`torch.cuda.max_memory_allocated()`，不包含CUDA驱动及PyTorch保留但未使用的内存，
也不是AGX整机内存。原图mask后处理和中间特征仍占主要显存。

### 是否满足“不影响结果”

FP32合并、固定文本相对同精度原P12的十图输出完全一致，这个结论仍成立。FP16有舍入和计算
误差，不能称为完全无损。相对FP32部署版，白实线预测前景差异约0.769%，白虚线约0.333%；
相对原P12 BF16，分别约4.03%和1.60%。这里前景差异为对称差像素数除以两版预测并集，
不是与真值的误差。背景很多的全图像素一致率会掩盖细线变化，因此同时记录两者。

相对原P12 BF16，10图中8张白实线和4张白虚线的候选数量发生变化。白实线Recall下降0.00444，
超过先前设定的0.002门槛；虽然平均IoU下降小于0.001，仍不能判定通过全部无损验收条件。
当前将FP16包列为速度/体积候选，保留FP32包作严格同精度参考，没有自动替换正式P12基线。

### 复现与单图命令

```bash
# 只转换一次；若已有同名包会拒绝覆盖
.venv/bin/python sam3_lightweight_p12_deploy_exp/precision_test.py convert
# 每条命令测试一个版本，可用CUDA_VISIBLE_DEVICES分配不同GPU
.venv/bin/python sam3_lightweight_p12_deploy_exp/precision_test.py fixed_fp32
.venv/bin/python sam3_lightweight_p12_deploy_exp/precision_test.py half_storage
.venv/bin/python sam3_lightweight_p12_deploy_exp/precision_test.py half_amp
.venv/bin/python sam3_lightweight_p12_deploy_exp/precision_test.py original_bf16
.venv/bin/python sam3_lightweight_p12_deploy_exp/summarize_precision.py

.venv/bin/python sam3_lightweight_p12_deploy_exp/infer.py \
  --package sam3_lightweight_p12_deploy_exp/weights/p12_fixed_vocab_fp16.pt \
  --precision fp16 --image /path/to/image.jpg \
  --prompt "white solid lane line" \
  --output sam3_lightweight_p12_deploy_exp/tests/output/fp16_prediction.png
```

完整结果为`tests/output/precision/summary.json`；`comparison_00.jpg`至`comparison_09.jpg`
依次展示原P12 BF16、FP16混合精度、差异区域。青色表示白实线/白虚线预测覆盖，橙色表示两版
覆盖不同，不代表人工确认的误检。NPZ保存逐类预测并集，便于复核。

下一步优先在更丰富的图片上复核关键细线Recall，并分别测试FP32权重保留关键层、固定BF16文本
特征等混合方案；空几何常量化及TensorRT仍作为后续独立消融。AGX实测尚未进行。
