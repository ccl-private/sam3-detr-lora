"""汇总四组精度测试，避免背景像素一致率掩盖细线变化。"""
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

ROOT=Path(__file__).resolve().parent/'tests/output/precision'
MODES=['fixed_fp32','half_storage','half_amp','original_bf16']
reports={m:json.loads((ROOT/m/'report.json').read_text()) for m in MODES}
summary={'reports':reports,'comparisons':{}}
for base,target in [('fixed_fp32','half_storage'),('fixed_fp32','half_amp'),('original_bf16','half_amp')]:
    changed=np.zeros(7,dtype=np.int64); union=changed.copy(); pixels=changed.copy(); counts=changed.copy()
    for i,item in enumerate(reports[base]['images']):
        assert item['path']==reports[target]['images'][i]['path']
        a=np.load(ROOT/base/f'{i}.npz')['masks']; b=np.load(ROOT/target/f'{i}.npz')['masks']
        changed+=(a!=b).sum((1,2)); union+=(a|b).sum((1,2)); pixels+=a.shape[1]*a.shape[2]
        counts+=np.array([x['detections']!=y['detections'] for x,y in zip(item['prompts'],reports[target]['images'][i]['prompts'])])
        if base=='original_bf16':
            original=Image.open(item['path']).convert('RGB')
            panels=[]
            for label,masks in [('original BF16',a),('fixed FP16 AMP',b),('difference',None)]:
                rgb=np.asarray(original).copy()
                select=(a[[0,2]].any(0)!=b[[0,2]].any(0)) if masks is None else masks[[0,2]].any(0)
                rgb[select]=(rgb[select]*0.4+np.array([255,140,0] if masks is None else [0,220,255])*0.6).astype(np.uint8)
                panel=Image.fromarray(rgb); panel.thumbnail((640,480))
                frame=Image.new('RGB',(640,520),'white'); frame.paste(panel,(0,35))
                ImageDraw.Draw(frame).text((10,10),label,fill='black'); panels.append(frame)
            canvas=Image.new('RGB',(1920,520),'white')
            for k,panel in enumerate(panels): canvas.paste(panel,(640*k,0))
            canvas.save(ROOT/f'comparison_{i:02d}.jpg')
    summary['comparisons'][f'{base}_to_{target}']={'changed_pixels':changed.tolist(),
        'foreground_disagreement':(changed/np.maximum(union,1)).tolist(),
        'all_pixel_agreement':(1-changed/pixels).tolist(),
        'images_with_detection_count_change':counts.tolist(),
        'mean_iou_delta':reports[target]['mean_iou_white']-reports[base]['mean_iou_white']}
(ROOT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
for mode,r in reports.items(): print(mode,r['storage_mib'],r['mean_iou_white'],r['timing'],r['peak_allocated_mib'])
print(json.dumps(summary['comparisons'],indent=2))
