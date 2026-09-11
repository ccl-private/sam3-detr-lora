"""P12固定词表单文件导出、加载、等价检查与测速。"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image
from torch.nn.utils import parametrize

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
EXP = ROOT / 'sam3_lightweight_tinyvit_stage3_distill_exp'
for path in (ROOT, ROOT / 'sam3_lightweight_stage3_exp', EXP,
             *[EXP / p for p in ('p5_dsconv_thin_line', 'p6_multiscale_dsconv',
                                'p7_highres_fpn', 'p8_input_line_branch', 'p9_fresh_p8_new_teacher')]):
    sys.path.insert(0, str(path))
from bootstrap import activate_efficientsam3
activate_efficientsam3()
import sam3.model_builder as builder
from sam3.model.sam3_image_processor import Sam3Processor
from model_adapter import build_trainable_stage3_detector
from complete_p8_structure import attach_complete_p8_structure
from text_feature_provider import PrecomputedTextEncoder

PROMPTS = ['white solid lane line', 'yellow solid lane line', 'white dashed lane line',
           'yellow dashed lane line', 'zebra crossing', 'lane barrier', 'road teeth marking']
BASE = ROOT / 'sam3_lightweight_stage3_exp/input/efficientsam3_tinyvit_stage3.pt'
LORA = EXP / 'weights/p12_query_set_distill.best.pt'
PACKAGE = HERE / 'weights/p12_fixed_vocab_fp32.pt'

def digest(path):
    with open(path, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def attach(model, meta):
    attach_complete_p8_structure(model, p5_branch_channels=meta['p5_branch_channels'],
        p6_branch_channels=meta['p6_stage1_branch_channels'], kernel_size=meta['p8_kernel_size'],
        offset_scale=meta['p8_offset_scale'], p8_operator=meta['p8_operator'],
        p8_stem_channels=meta['p8_stem_channels'], p8_line_channels=meta['p8_line_channels'])

def reference(base=BASE, lora=LORA):
    payload = torch.load(lora, map_location='cpu', weights_only=False)
    meta = payload['meta']
    keys = ('lora_rank', 'lora_alpha', 'lora_dropout', 'decoder_only', 'attn_only',
            'train_dot_score', 'train_seg_head', 'image_lora_rank', 'image_lora_alpha',
            'image_lora_dropout', 'image_lora_stages')
    model, _ = build_trainable_stage3_detector(checkpoint_path=base, text_mode='runtime',
        text_cache_path=None, **{k: meta[k] for k in keys})
    attach(model, meta)
    missing, unexpected = model.load_state_dict(payload['state_dict'], strict=False)
    relevant = [k for k in missing if 'parametrizations' in k or k.startswith(
        ('p5_', 'p6_', 'p7_', 'p8_', 'dot_prod_scoring.', 'segmentation_head.'))]
    if relevant or unexpected:
        raise RuntimeError((relevant, unexpected))
    return model.eval(), meta

def load_package(path=PACKAGE):
    p = torch.load(path, map_location='cpu', weights_only=True)
    if p['format'] != 'p12_fixed_v1':
        raise ValueError('不支持的部署包')
    encoder = PrecomputedTextEncoder(p['prompts'], **p['text'])
    # 仅在构建期间替换工厂；不创建MobileCLIP，不加载原始基模或LoRA文件。
    with patch.object(builder, '_create_text_encoder', return_value=encoder):
        model = builder.build_efficientsam3_image_model(device='cpu', load_from_HF=False,
            backbone_type='tinyvit', model_name='11m', enable_inst_interactivity=False)
    attach(model, p['structure'])
    model.load_state_dict(p['state_dict'], strict=True)
    return model.eval()

@torch.inference_mode()
def export(args):
    model, meta = reference(args.base, args.lora)
    model.to(args.device)
    # 按单提示独立计算，与Sam3Processor实际使用方式保持一致。
    values = [model.backbone.language_backbone([p], device=args.device) for p in PROMPTS]
    text = dict(attention_masks=torch.cat([v[0] for v in values], 0).cpu(),
                text_memories=torch.cat([v[1] for v in values], 1).cpu(),
                text_embeds=torch.cat([v[2] for v in values], 1).cpu())
    model.cpu()
    merged = []
    for name, module in list(model.named_modules()):
        if hasattr(module, 'parametrizations'):
            for key in list(module.parametrizations.keys()):
                parametrize.remove_parametrizations(module, key, leave_parametrized=True)
                merged.append(name + '.' + key)
    model.backbone.language_backbone = PrecomputedTextEncoder(PROMPTS, **text)
    state = {k: v.detach().cpu().clone() for k,v in model.state_dict().items()}
    args.package.parent.mkdir(parents=True, exist_ok=True)
    if args.package.exists():
        raise FileExistsError(f'避免覆盖已有产物：{args.package}')
    torch.save(dict(format='p12_fixed_v1', prompts=PROMPTS, text=text, structure=meta,
        state_dict=state, sources={str(args.base): digest(args.base), str(args.lora): digest(args.lora)},
        merged_parameters=merged), args.package)
    loaded = load_package(args.package)
    assert all(torch.equal(v, loaded.state_dict()[k]) for k,v in state.items())
    print(json.dumps(dict(package=str(args.package), mib=args.package.stat().st_size/2**20,
        merged=len(merged), parameters=sum(p.numel() for p in loaded.parameters()),
        reload_exact=True), indent=2), flush=True)

def sync():
    if torch.cuda.is_available(): torch.cuda.synchronize()

def run(processor, image, prompts):
    state = processor.set_image(image)
    for prompt in prompts:
        processor.set_text_prompt(prompt, state)
    return state

@torch.inference_mode()
def verify(args):
    original, _ = reference(args.base, args.lora)
    deployed = load_package(args.package)
    models = [original.to(args.device).eval(), deployed.to(args.device).eval()]
    processors = [Sam3Processor(m, device=args.device, confidence_threshold=0.5) for m in models]
    manifest = json.loads(args.manifest.read_text())
    images = [Path(p) for p in manifest['images']][:args.limit]
    report = {'device':torch.cuda.get_device_name(), 'precision':'fp32', 'images':[], 'timing':{},
              'warmup':args.warmup, 'repeats':args.repeats, 'package_sha256':digest(args.package)}
    for path in images:
        image = Image.open(path).convert('RGB')
        outputs = []
        for model, proc in zip(models, processors):
            raw = []
            forward = model.forward_grounding
            def capture(*a, **kw):
                out = forward(*a, **kw)
                raw.append({k:out[k].detach().cpu() for k in
                    ('pred_logits', 'presence_logit_dec', 'pred_boxes', 'pred_masks')})
                return out
            model.forward_grounding = capture
            state = proc.set_image(image)
            results = []
            for prompt in PROMPTS:
                out = proc.set_text_prompt(prompt, state)
                results.append({k:out[k].detach().cpu() for k in ('masks', 'boxes', 'scores')})
            model.forward_grounding = forward
            outputs.append((raw, results))
        rows = []
        for i,prompt in enumerate(PROMPTS):
            a,b = outputs[0][1][i], outputs[1][1][i]
            rows.append(dict(prompt=prompt, raw_max_abs={k:float((outputs[0][0][i][k]-outputs[1][0][i][k]).abs().max())
                 for k in outputs[0][0][i]}, detections=[len(a['scores']),len(b['scores'])],
                 masks_equal=torch.equal(a['masks'],b['masks']),
                 mask_changed_pixels=int((a['masks'] != b['masks']).sum()) if a['masks'].shape==b['masks'].shape else None))
        report['images'].append(dict(path=str(path), comparisons=rows))
        print(path.name, 'mask exact=',all(r['masks_equal'] for r in rows), flush=True)
    image = Image.open(images[0]).convert('RGB')
    for label, proc in zip(('reference','fixed_merged'), processors):
        for n in (1,7):
            for _ in range(args.warmup): run(proc,image,PROMPTS[:n])
            durations=[]
            for _ in range(args.repeats):
                sync(); start=time.perf_counter(); run(proc,image,PROMPTS[:n]); sync()
                durations.append((time.perf_counter()-start)*1000)
            report['timing'][f'{label}_{n}_prompts'] = dict(mean_ms=float(np.mean(durations)),
                p50_ms=float(np.percentile(durations,50)),p95_ms=float(np.percentile(durations,95)))
    report['all_masks_equal']=all(r['masks_equal'] for im in report['images'] for r in im['comparisons'])
    report['raw_max_abs']={k:max(r['raw_max_abs'][k] for im in report['images'] for r in im['comparisons'])
        for k in ('pred_logits','presence_logit_dec','pred_boxes','pred_masks')}
    report['raw_tolerance_pass']=all(v <= (1e-4 if k=='pred_masks' else 1e-5)
        for k,v in report['raw_max_abs'].items())
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps(report['timing'],indent=2), flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['export','verify'])
    p.add_argument('--base',type=Path,default=BASE); p.add_argument('--lora',type=Path,default=LORA)
    p.add_argument('--package',type=Path,default=PACKAGE); p.add_argument('--device',default='cuda:0')
    p.add_argument('--manifest',type=Path,default=EXP/'tests/output/p12_query_set_best_first10_threshold_05/summary.json')
    p.add_argument('--output',type=Path,default=HERE/'tests/output/fp32_report.json')
    p.add_argument('--limit',type=int,default=10); p.add_argument('--warmup',type=int,default=5)
    p.add_argument('--repeats',type=int,default=20)
    args=p.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    (export if args.action=='export' else verify)(args)

if __name__=='__main__': main()
