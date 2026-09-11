"""移除配对Q/DQ生成未量化FP16对照，仅适用于仍保留原浮点权重的模型。"""
import argparse
from pathlib import Path

import onnx
import numpy as np


def main(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    model = onnx.load(args.input)
    producers = {name: node for node in model.graph.node for name in node.output}
    constants = {value.name: onnx.numpy_helper.to_array(value) for value in model.graph.initializer}
    aliases = {}
    removed = set()
    for node in model.graph.node:
        if node.op_type != 'DequantizeLinear':
            continue
        quantizer = producers.get(node.input[0])
        if quantizer is None or quantizer.op_type != 'QuantizeLinear':
            raise ValueError('原浮点权重不可恢复：DQ未配对Q')
        if len(quantizer.input) != len(node.input) or not all(
            a == b or (a in constants and b in constants
                       and constants[a].dtype == constants[b].dtype
                       and np.array_equal(constants[a], constants[b]))
            for a, b in zip(quantizer.input[1:], node.input[1:])
        ):
            raise ValueError('Q/DQ参数不匹配')
        aliases[node.output[0]] = quantizer.input[0]
        removed.update((node.name, quantizer.name))
    nodes = []
    for node in model.graph.node:
        if node.name in removed:
            continue
        for index, name in enumerate(node.input):
            while name in aliases:
                name = aliases[name]
            node.input[index] = name
        nodes.append(node)
    if any(value.name in aliases for value in model.graph.output):
        raise ValueError('不支持直接以DQ作为模型输出')
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    # 删除失去生产者的中间类型声明；保留原浮点权重，检查推断后的数据类型。
    info = [value for value in model.graph.value_info if value.name not in aliases
            and not (value.name in producers and producers[value.name].name in removed)]
    del model.graph.value_info[:]
    model.graph.value_info.extend(info)
    onnx.checker.check_model(model, full_check=True)
    onnx.save(model, args.output)
    print(f'Removed {len(aliases)} Q/DQ pairs; full ONNX check passed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())
