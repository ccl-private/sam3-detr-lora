# 固定白实线 ONNX / TensorRT 部署记录

本文是固定`white solid lane line`单提示的ONNX/TensorRT实验记录。范围是单图、1008×1008输入，
输出200个候选的分类、Presence、框和mask logits。**不能将此实验的结果推广到全部7个提示或AGX实机。**

PyTorch部署包支持全部7个提示，见[部署实验README](README.md)；选型和入口见
[轻量化部署最佳实践](BEST_PRACTICES.md)。

## 1. 当前结论

白实线精度取自[部署实验README](README.md)（PyTorch两行）与本页3.3节（TensorRT两行）；
延迟为本页实测，GPU张量口径：网络前向见3.3节，端到端见第4节。

| 方案 | 白实线IoU | 白实线Recall | 网络前向 | 端到端组合实测 |
|---|---:|---:|---:|---:|
| PyTorch FP32部署包（严格基准） | 0.654836 | 0.714868 | 未测 | 未测 |
| PyTorch FP16 autocast | 0.654737 | 0.714003 | 61.63 ms | 63.50 ms |
| **TensorRT FP16单提示引擎** | 0.648045 | 0.709623 | **22.68 ms** | **24.74 ms** |
| TensorRT INT8引擎 | 0.388614 | 0.437857 | 23.40 ms | 未测 |

- **推荐路径**：`export_fp16_onnx.py`直转FP16 → `build_tensorrt.py`构建引擎 → 端到端24.74 ms。
  不需要INT8量化，也不需要剥离Q/DQ；引擎精度与ModelOpt路线等价且略好。
  推理入口见`infer_tensorrt.py`。
- **INT8已排除**：两种运行时下精度都不可用（IoU 0.39～0.50），相对FP16也没有延迟收益，
  只多付302秒构建时间和222对Q/DQ。
- **TensorRT引擎把提示写死在图里**：一个引擎只对应一个提示，换提示必须重新导出；
  这与PyTorch部署包的7类固定词表不同。
- 未完成：**AGX实机未构建、未测**；**7个提示在TensorRT上的代价未测**（7个引擎串行估算约159 ms，
  可能慢于PyTorch批处理路径的149 ms）。因此本页给不出"全部7类的最优路径"。

本页所有延迟都是A800本机数字，不能推算AGX帧率。

## 2. 产物与可复现命令

### 2.1 推荐路径

```bash
# 从FP32 ONNX直转FP16
.venv/bin/python sam3_lightweight_p12_deploy_exp/export_fp16_onnx.py \
  --input sam3_lightweight_p12_deploy_exp/weights/p12_white_solid_trt_fp32.onnx \
  --output sam3_lightweight_p12_deploy_exp/weights/p12_white_solid_trt_fp16_direct.onnx
# 构建本机引擎，关闭TF32
.venv/bin/python sam3_lightweight_p12_deploy_exp/build_tensorrt.py \
  --onnx sam3_lightweight_p12_deploy_exp/weights/p12_white_solid_trt_fp16_direct.onnx \
  --output sam3_lightweight_p12_deploy_exp/weights/p12_white_solid_trt_fp16_direct.engine
```

`export_fp16_onnx.py`用onnxruntime自带的FP16转换器，转换器的默认屏蔽表不含Resize，
只把Range等少数算子留在FP32，与ModelOpt转换后的精度分布一致。脚本另做两件必要处理：

- ORT转换器把新插入的输入Cast追加在图的末尾，需要重新拓扑排序才能通过检查。
- 转换器把空的Resize roi/scales常量包进Cast，ONNX形状推断因此无法再按常量值判定scales为空，
  会报"sizes与scales不能同时提供"；脚本沿Cast回溯源常量，删除空scales输入并清理失去消费者的Cast。

FP32 ONNX由`export_onnx.py`从[部署实验README](README.md)第2节导出的FP32部署包产生。
`--tensorrt-compatible`关闭P8 Resize抗锯齿（TensorRT不支持），这会改变mask输出，
因此第3.1节用`evaluate_onnx.py`单独核对了精度；不加该开关得到的是保留抗锯齿的
`p12_white_solid_static_fp32.onnx`。两个文件名都不能改，后续脚本按这些名字引用。

```bash
.venv/bin/python sam3_lightweight_p12_deploy_exp/export_onnx.py \
  --tensorrt-compatible \
  --output sam3_lightweight_p12_deploy_exp/weights/p12_white_solid_trt_fp32.onnx
```

`build_tensorrt.py`解析ONNX并保存构建JSON，固定清除TF32标志；构建耗时约250秒。

### 2.2 评估与测速

```bash
# 精度：10图对照PyTorch，输出原始误差、mask差异、白实线IoU与Recall
.venv/bin/python sam3_lightweight_p12_deploy_exp/evaluate_onnx.py \
  --model sam3_lightweight_p12_deploy_exp/weights/p12_white_solid_trt_fp16_direct.engine \
  --output sam3_lightweight_p12_deploy_exp/tests/output/onnx_int8/evaluate_engine_fp16_direct.json \
  --repeats 50
# 端到端分项测速（预处理／引擎前向／后处理）
.venv/bin/python sam3_lightweight_p12_deploy_exp/benchmark_engine_stages.py \
  --engine sam3_lightweight_p12_deploy_exp/weights/p12_white_solid_trt_fp16_direct.engine \
  --warmup 20 --repeats 100
# 单图推理
.venv/bin/python sam3_lightweight_p12_deploy_exp/infer_tensorrt.py \
  --engine sam3_lightweight_p12_deploy_exp/weights/p12_white_solid_trt_fp16_direct.engine \
  --image /path/to/image.png \
  --output sam3_lightweight_p12_deploy_exp/tests/output/onnx_int8/prediction.png
```

`evaluate_onnx.py`检查有限值、原始输出误差、原图白实线union-mask IoU和Recall；ONNX Runtime
关闭TF32。`infer_tensorrt.py`是独立单图引擎入口，不加载原PyTorch模型权重或训练代码，
保存覆盖图与同名NPZ（实例mask、原图坐标框、分数）。两者仍需本机兼容的TensorRT、PyTorch、
torchvision、NumPy和Pillow。

### 2.3 已排除：INT8路径

保留命令供复核。量化策略为Conv/MatMul INT8，排除P5/P6/P8细线分支、分割头和dot-product评分头，
其他高精度计算使用FP16；**排除INT8不代表保留FP32**。校准图与10张验证图按真实文件路径检查
无重叠，未检查视频级重叠。

```bash
# 32张训练图校准数组；ModelOpt保守INT8量化
.venv/bin/python sam3_lightweight_p12_deploy_exp/prepare_int8_calibration.py
bash sam3_lightweight_p12_deploy_exp/quantize_int8.sh
# ModelOpt输出需先修复Resize可选参数，再构建引擎
.venv/bin/python sam3_lightweight_p12_deploy_exp/fix_quantized_resize.py \
  --input sam3_lightweight_p12_deploy_exp/weights/p12_white_solid_trt_int8_conservative.onnx \
  --output sam3_lightweight_p12_deploy_exp/weights/p12_white_solid_trt_int8_conservative_fixed.onnx
.venv/bin/python sam3_lightweight_p12_deploy_exp/build_tensorrt.py \
  --onnx sam3_lightweight_p12_deploy_exp/weights/p12_white_solid_trt_int8_conservative.onnx \
  --output sam3_lightweight_p12_deploy_exp/weights/p12_white_solid_trt_int8_conservative.engine
```

另有两份对照产物：

```bash
# 从同一模型移除222对Q/DQ，保留原浮点权重与FP16转换结果
.venv/bin/python sam3_lightweight_p12_deploy_exp/make_fp16_control.py \
  --input sam3_lightweight_p12_deploy_exp/weights/p12_white_solid_trt_int8_conservative.onnx \
  --output sam3_lightweight_p12_deploy_exp/weights/p12_white_solid_trt_fp16_control.onnx
# INT8在ONNX Runtime下的精度核对（3.2节的IoU）
.venv/bin/python sam3_lightweight_p12_deploy_exp/evaluate_onnx.py \
  --model sam3_lightweight_p12_deploy_exp/weights/p12_white_solid_trt_int8_conservative_fixed.onnx \
  --output sam3_lightweight_p12_deploy_exp/tests/output/onnx_int8/evaluate_onnx_int8_fixed.json
```

`p12_white_solid_trt_int8_fp32.onnx`（非量化层保留FP32）是同一量化命令改用opset19中间文件
`p12_white_solid_trt_fp32_opset19.onnx`得到的，该中间文件已不在仓库中，因此这一份只能作为
历史记录，不能按现有命令重建。

## 3. 精度结果

### 3.1 导出与TF32基准

10图真实标签白实线IoU（像素累计后计算），置信度阈值0.5：

| 版本 | 产物 | IoU |
|---|---|---:|
| 原PyTorch FP32，关闭TF32 | FP32部署包 | 0.654836 |
| ONNX保留抗锯齿，ORT关闭TF32 | `p12_white_solid_static_fp32.onnx` | 0.638322 |
| TensorRT兼容ONNX关闭抗锯齿，ORT关闭TF32 | `p12_white_solid_trt_fp32.onnx` | 0.655663 |
| TensorRT兼容ONNX，ORT默认TF32 | 同上 | 0.670094 |
| 首次TensorRT FP32引擎，默认TF32 | `p12_white_solid_trt_fp32.engine` | 0.670033 |
| TensorRT FP32引擎，关闭TF32 | `p12_white_solid_trt_fp32_no_tf32.engine` | 0.655666 |

保留抗锯齿的ONNX也有mask差异，尚不能称为等价导出。严格FP32下分类/框误差显著缩小，
但mask差异仍存在。**不能用默认TF32的IoU变化代表INT8量化收益。**
所有逐图误差和检测数量见`tests/output/onnx_int8/evaluate_*.json`。

### 3.2 INT8结果

原INT8＋FP16量化于14:54:31完成；ONNX为151.544MiB，222对Q/DQ、121个量化节点。

| ONNX Runtime对照 | 10图白实线IoU |
|---|---:|
| INT8＋其余层FP16，修复Resize后 | 0.502197 |
| INT8＋其余层FP32 | 0.428385 |
| 从同一FP16模型移除222对Q/DQ | 0.643246 |

当前INT8参数导致明显精度下降，不能作为推荐替换。FP16对照恢复了大部分精度，但仍不等价于原FP32。
`make_fp16_control.py`要求每个DQ都有对应Q及相同的量化参数，只绕过Q/DQ，保留原浮点权重与
FP16转换结果；生成后通过完整ONNX检查。

### 3.3 引擎级验证

各引擎此前只完成构建，2026-09-11补测。每轮一个进程占一张A800，10图、阈值0.5、关闭TF32、
预热10次后重复50次。`evaluate_onnx.py`的Recall口径与`precision_test.py`相同
（像素累计后按真值求召回）。

| 引擎 | 构建秒 | 引擎MiB | 权重MiB | 白实线IoU | 白实线Recall | 平均ms | P50 | P95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FP32（TF32开） | 128.30 | 1060.83 | 1053.2 | 0.670033 | 0.726976 | 42.156 | 42.083 | 42.703 |
| FP32无TF32 | 112.47 | 1096.63 | 1090.5 | 0.655666 | 0.710539 | 68.992 | 68.969 | 69.252 |
| INT8＋FP16 | 302.44 | 777.96 | 767.7 | 0.388614 | 0.437857 | 23.401 | 23.393 | 23.501 |
| FP16对照（移除222对Q/DQ） | 258.62 | 746.33 | 736.9 | 0.647420 | 0.708803 | 22.836 | 22.835 | 22.870 |
| FP16直转 | 249.99 | 1029.84 | 1020.4 | 0.648045 | 0.709623 | 22.675 | 22.677 | 22.706 |

PyTorch FP32基准为IoU 0.654836、Recall 0.714868，与历史FP32部署包的Recall 0.714868一致。
构建秒取`.build.json`的`elapsed_seconds`，与构建日志中TensorRT自报的引擎生成时间口径不同
（如首次INT8引擎TRT自报298.52秒）；权重MiB取构建日志的Total Weights Memory。
重复运行同一引擎时IoU逐位相同，平均耗时波动约1%。测速范围为GPU归一化输入到GPU原始输出，
含输入复制与输出clone，不含图像预处理、原图mask后处理、图片解码，**不能与包含预处理/后处理
的历史结果比较**。文件名中的FP32也不能替代实际TF32配置：同一份FP32 ONNX在TF32开／关下分别是
42.156ms和68.992ms，精度也不同。

- INT8引擎IoU 0.388614、Recall 0.437857，比同一模型在ONNX Runtime上的IoU 0.502197更低；
  10张图的候选数量全部与PyTorch不同，逐图成对mask IoU最低0.2479。INT8在两种运行时下都不可用。
  该引擎由未修复的原ONNX构建，两种ONNX语义相同，但实际执行的量化内核不同。
- INT8引擎相对严格FP32引擎提速2.95倍，FP16对照引擎同样达到3.02倍且IoU高0.2588。
  本批图上INT8没有带来相对FP16的延迟收益。
- FP16对照引擎相对PyTorch基准IoU低0.007416、Recall低0.006065；相对**同一导出的FP32引擎**
  IoU低0.008246、Recall低0.001736。FP32引擎本身相对PyTorch已低0.004329 Recall，因此按Recall
  判断时导出路径的偏差大于FP16转换，两者不能合并成一个"FP16损失"。
- FP16对照引擎10图中有7张的候选数量与PyTorch不同（相差1～3个），逐图成对mask IoU为
  0.8410～0.9562；FP32引擎候选数10/10一致、逐图mask IoU为0.9434～0.9543。
- 移除Q/DQ前后的ONNX初始化器集合完全相同（444个INT8零点、1449个FP16共142.8MiB；
  节点数13146→12702），说明FP16对照用的是转换后的原FP16权重，没有重新量化。

### 3.4 FP16直转与ModelOpt路线对照

同一ONNX Runtime下的模型对照，用于隔离转换差异：

| ONNX | 白实线IoU | 白实线Recall |
|---|---:|---:|
| ModelOpt路线fp16_control | 0.643246 | 0.702082 |
| 直转fp16_direct | 0.643142 | 0.701885 |

引擎结果见上表：直转IoU 0.648045、Recall 0.709623、平均22.675ms，三项都略优于ModelOpt路线。
两个FP16引擎的Total Activation Memory完全相同（413,295,104字节），reformat层数也相同（65），
说明运行时计算图一致。两条路径保留的FP32岛不同：ModelOpt转换后只有8个回FP32的Cast，全部给Range；
直转有22个，涉及Range和CumSum（CumSum也在转换器的默认屏蔽表内）。

## 4. 单提示端到端

`benchmark_engine_stages.py`用与`benchmark_latency_stages.py`相同的独立同步计时，把直转引擎的
单提示链路拆成预处理、引擎前向和后处理。下表PyTorch列为正式记录值
（`latency_stages/report.json`）；同一次会话内重测为0.77／61.21／1.41／63.03ms，差异约1%：

| 阶段 | TensorRT FP16引擎 | PyTorch FP16 autocast |
|---|---:|---:|
| 预处理（上游GPU uint8张量） | 0.78 ms | 0.78 ms |
| 网络前向 | 22.68 ms | 61.63 ms |
| 后处理（阈值、框变换、mask上采样到原图） | 1.28 ms | 1.41 ms |
| **组合链路实测** | **24.74 ms（40.4张/秒）** | **63.50 ms（15.7张/秒）** |

同一张3840×2160图、单提示、batch 1、预热20次、重复100次，不含磁盘读图、图片解码和引擎反序列化。
引擎路径的预处理和后处理在FP32下运行，没有PyTorch路径的FP16 autocast。
分项之和0.78＋22.68＋1.28＝24.74与组合实测相同，说明各阶段边界开销可以忽略。

引擎前向占端到端延时的91.7%，因此这条链路的进一步优化只能来自网络本身。结果为
`tests/output/latency_stages/engine_report.json`。

## 5. 已知问题

- **Resize可选参数**：ModelOpt输出的P8 Resize空scales被转为FP16，ORT报INVALID_GRAPH。
  `fix_quantized_resize.py`将已经提供sizes时的空scales改为缺省参数，未修改权重；
  修复后通过ONNX完整类型检查。首次INT8引擎由原ONNX构建，TensorRT解析器接受该空参数，
  ORT使用修复后的等义模型。
- **直转引擎体积未解释**：直转引擎文件1029.84MiB、权重1020.4MiB，比ModelOpt路线的
  746.33MiB、736.9MiB大约300MB；而直转ONNX的FP16初始化器只有109.7MiB（正好是FP32源
  219.4MiB的一半），比ModelOpt的142.8MiB更少。激活内存与延迟都没有相应变化，原因尚未定位。
  按引擎体积选择时FP16对照更优；按流程简洁度选择时直转只需两条命令。

## 6. 限制

本页所有结论限于固定白实线单提示、10张验证图、A800本机构建物，不能推广到其余6个提示。
AGX需在目标机重新构建引擎、验证和测速。
