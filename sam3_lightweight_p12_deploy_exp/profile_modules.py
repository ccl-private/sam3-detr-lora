"""固定文本FP16推理分模块计时；嵌套区间不可重复相加。"""
import json
import time
from collections import defaultdict
from functools import wraps
import torch
from PIL import Image
from deploy import HERE, EXP, PROMPTS, load_package, Sam3Processor, run

@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = load_package(HERE/'weights/p12_fixed_vocab_fp16.pt').float().cuda().eval()
    proc = Sam3Processor(model, device='cuda:0', confidence_threshold=0.5)
    manifest = EXP/'tests/output/p12_query_set_best_first10_threshold_05/summary.json'
    path = json.loads(manifest.read_text())['images'][0]
    image = Image.open(path).convert('RGB')
    output = HERE/'tests/output/module_profile'
    output.mkdir(parents=True, exist_ok=True)
    records = defaultdict(list)
    active = False
    tracing = False
    def wrap(obj, attr, name):
        old = getattr(obj, attr)
        @wraps(old)
        def measured(*args, **kwargs):
            if tracing:
                with torch.profiler.record_function(name):
                    return old(*args, **kwargs)
            if not active:
                return old(*args, **kwargs)
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            result = old(*args, **kwargs)
            end.record()
            records[name].append((start, end))
            return result
        setattr(obj, attr, measured)
    for obj, attr, name in [(proc,'set_image','processor.image'),
                             (proc,'set_text_prompt','processor.prompt'),
                             (model,'forward_grounding','grounding'),
                             (model.backbone,'forward_image','backbone.image')]:
        wrap(obj, attr, name)
    selected = ('transformer.encoder','transformer.decoder','segmentation_head',
                'dot_prod_scoring','geometry_encoder','p5_thin_line_branch',
                'p6_stage1_thin_line_branch','p7_highres_fpn_adapters','p8_input_line_branch')
    for name, module in model.named_modules():
        if name in selected or type(module).__name__ in ('TinyViT','DynamicSnakeConv2d'):
            wrap(module, 'forward', name)
    report = {'gpu':torch.cuda.get_device_name(), 'image':path, 'size':image.size,
              'precision':'FP16 autocast, FP32 parameters', 'warmup':20, 'repeats':50,
              'note':'CUDA事件为包含子调用的流时间（可能含CPU发射间隙）；嵌套不可相加。算子表为独立profiler采样。', 'timing':{}}
    with torch.autocast('cuda', dtype=torch.float16):
        for n in (1,7):
            for _ in range(20): run(proc,image,PROMPTS[:n])
            times=[]
            for _ in range(50):
                torch.cuda.synchronize(); begin=time.perf_counter()
                run(proc,image,PROMPTS[:n]); torch.cuda.synchronize()
                times.append((time.perf_counter()-begin)*1000)
            records.clear(); active=True
            for _ in range(50): run(proc,image,PROMPTS[:n])
            torch.cuda.synchronize(); active=False
            modules={k:{'ms_per_image':sum(s.elapsed_time(e) for s,e in v)/50,
                         'calls_per_image':len(v)/50} for k,v in records.items()}
            report['timing'][str(n)]={'uninstrumented_wall_ms':sum(times)/50,'modules':modules}
            print(n,json.dumps(report['timing'][str(n)]),flush=True)
        tracing=True
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                               torch.profiler.ProfilerActivity.CUDA]) as prof:
            for _ in range(3): run(proc,image,PROMPTS[:1])
            torch.cuda.synchronize()
        tracing=False
    operators=[]
    for event in prof.key_averages():
        # 只保留CPU侧ATen算子归属，排除CUDA内核与用户范围，避免重复统计。
        if not event.key.startswith('aten::'):
            continue
        operators.append({'name':event.key,'self_gpu_ms_per_image':event.self_device_time_total/3000,
                          'inclusive_gpu_ms_per_image':event.device_time_total/3000,
                          'self_cpu_ms_per_image':event.self_cpu_time_total/3000,'calls_per_image':event.count/3})
    report['operators']=sorted(operators,key=lambda x:x['self_gpu_ms_per_image'],reverse=True)
    prof.export_chrome_trace(str(output/'trace.json'))
    (output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps(report['operators'][:20],ensure_ascii=False,indent=2),flush=True)

if __name__ == '__main__':
    main()
