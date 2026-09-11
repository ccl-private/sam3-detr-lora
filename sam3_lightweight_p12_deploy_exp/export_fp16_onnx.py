"""从FP32固定形状ONNX直接转换FP16，用于验证能否跳过ModelOpt量化流程。"""
import argparse
from pathlib import Path

import onnx
from onnxruntime.transformers import float16


def toposort(graph):
    """按依赖重排节点；ORT转换器会把新插入的输入Cast追加在图的末尾。"""
    index = {node.name: position for position, node in enumerate(graph.node)}
    producer = {output: node.name for node in graph.node for output in node.output}
    consumers = {node.name: set() for node in graph.node}
    indegree = {node.name: 0 for node in graph.node}
    for node in graph.node:
        for name in node.input:
            source = producer.get(name)
            if source is None or source == node.name or node.name in consumers[source]:
                continue
            consumers[source].add(node.name)
            indegree[node.name] += 1
    by_name = {node.name: node for node in graph.node}
    ready = [name for name in index if indegree[name] == 0]
    ordered = []
    while ready:
        ready.sort(key=index.get)
        name = ready.pop(0)
        ordered.append(by_name[name])
        for child in consumers[name]:
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)
    if len(ordered) != len(index):
        raise ValueError('计算图存在环，无法拓扑排序')
    del graph.node[:]
    graph.node.extend(ordered)


def drop_empty_scales(graph):
    """删除Resize已提供sizes时的空scales输入，并清理因此失去消费者的Cast。

    转换器会把空的roi/scales常量包进Cast，ONNX形状推断无法再按常量值判定scales为空，
    会报sizes与scales不能同时提供。这里沿Cast回溯源常量识别这种情况。
    """
    producer = {output: node for node in graph.node for output in node.output}
    constants = {node.output[0]: node for node in graph.node if node.op_type == 'Constant'}
    nodes = list(graph.node)
    dropped = 0
    for node in nodes:
        if node.op_type != 'Resize' or len(node.input) < 4 or not node.input[2] or not node.input[3]:
            continue
        source = producer.get(node.input[2])
        if source is not None and source.op_type == 'Cast':
            source = constants.get(source.input[0])
        value = None if source is None else next(
            (attribute.t for attribute in source.attribute if attribute.name == 'value'), None)
        if value is not None and 0 in value.dims:
            node.input[2] = ''
            dropped += 1
    used = {name for node in nodes for name in node.input} | {value.name for value in graph.output}
    nodes = [node for node in nodes if node.op_type != 'Cast' or node.output[0] in used]
    del graph.node[:]
    graph.node.extend(nodes)
    return dropped


def main(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    model = onnx.load(args.input)
    # keep_io_types保留FP32输入输出，与已有引擎接口一致。转换器的默认屏蔽表不含Resize，
    # 只把Range等少数算子留在FP32，与ModelOpt转换后的精度分布一致。
    converted = float16.convert_float_to_float16(model, keep_io_types=True)
    print(f'Dropped {drop_empty_scales(converted.graph)} empty Resize scales; full ONNX check follows')
    toposort(converted.graph)
    onnx.checker.check_model(converted, full_check=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(converted, args.output)
    print(f'FP16 ONNX written: {args.output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())
