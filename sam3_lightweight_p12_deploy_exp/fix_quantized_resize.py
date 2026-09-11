"""修复ModelOpt FP16转换后Resize scales被错误转换为FP16的问题。"""
import argparse
from pathlib import Path

import onnx
from onnx import helper, TensorProto


def main(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    model = onnx.load(args.input)
    names = {name for node in model.graph.node for name in (*node.input, *node.output)}
    constants = {value.name: value for value in model.graph.initializer}
    for node in model.graph.node:
        if node.op_type == 'Constant':
            for attribute in node.attribute:
                if attribute.name == 'value':
                    constants[node.output[0]] = attribute.t
    nodes = []
    count = 0
    for index, node in enumerate(model.graph.node):
        # ONNX Resize的scales类型固定为tensor(float)，不随图像数据类型改变。
        if node.op_type == 'Resize' and len(node.input) > 2 and node.input[2]:
            value = constants.get(node.input[2])
            if value is not None and 0 in value.dims and len(node.input) > 3 and node.input[3]:
                node.input[2] = ''
                nodes.append(node)
                count += 1
                continue
            name = f'p12_resize_scales_fp32_{index}'
            if name in names:
                raise ValueError(f'Name collision: {name}')
            nodes.append(helper.make_node('Cast', [node.input[2]], [name],
                                          name=name, to=TensorProto.FLOAT))
            node.input[2] = name
            count += 1
        nodes.append(node)
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    onnx.checker.check_model(model, full_check=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, args.output)
    print(f'Fixed {count} Resize scales inputs; full ONNX check passed: {args.output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())
