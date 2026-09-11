"""固定形状TensorRT引擎运行器，不加载训练代码或PyTorch模型权重。"""
import numpy as np
import torch

NAMES = ['pred_logits', 'presence_logit_dec', 'pred_boxes', 'pred_masks']


class EngineRunner:
    def __init__(self, path):
        import tensorrt as trt
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(path.read_bytes())
        if self.engine is None:
            raise RuntimeError('引擎反序列化失败')
        self.context = self.engine.create_execution_context()
        self.buffers = {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            dtype = torch.from_numpy(np.empty((), dtype=trt.nptype(self.engine.get_tensor_dtype(name)))).dtype
            self.buffers[name] = torch.empty(tuple(self.engine.get_tensor_shape(name)), dtype=dtype, device='cuda')
            self.context.set_tensor_address(name, self.buffers[name].data_ptr())

    def __call__(self, image):
        self.buffers['image'].copy_(image)
        if not self.context.execute_async_v3(torch.cuda.current_stream().cuda_stream):
            raise RuntimeError('TensorRT推理失败')
        return [self.buffers[name].clone() for name in NAMES]

