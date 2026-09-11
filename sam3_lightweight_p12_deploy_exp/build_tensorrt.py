"""从显式精度ONNX构建本机TensorRT引擎，保存解析错误及构建信息。"""
import argparse
import json
import time
from pathlib import Path

import tensorrt as trt


def main(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    logger = trt.Logger(trt.Logger.INFO)
    trt.init_libnvinfer_plugins(logger, "")
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)
    report = {"onnx": str(args.onnx), "engine": str(args.output),
              "tensorrt": trt.__version__, "status": "parsing"}
    report_path = args.output.with_suffix(".build.json")
    start = time.perf_counter()
    try:
        if not parser.parse_from_file(str(args.onnx)):
            report["parser_errors"] = [str(parser.get_error(i)) for i in range(parser.num_errors)]
            raise RuntimeError("TensorRT ONNX解析失败，详见构建报告")
        config = builder.create_builder_config()
        if hasattr(trt.BuilderFlag, 'TF32'):
            config.clear_flag(trt.BuilderFlag.TF32)
        report['tf32_enabled'] = False
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, args.workspace_gib * 1024**3)
        engine = builder.build_serialized_network(network, config)
        if engine is None:
            raise RuntimeError("TensorRT引擎构建失败，详见控制台日志")
        args.output.write_bytes(bytes(engine))
        report.update(status="built", engine_mib=args.output.stat().st_size / 2**20)
    except Exception as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        report["elapsed_seconds"] = time.perf_counter() - start
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workspace-gib", type=int, default=8)
    main(parser.parse_args())
