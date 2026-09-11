"""导出固定白实线、单提示、固定1008输入的纯神经网络ONNX。"""
import argparse
import json
from pathlib import Path
import torch
from torch import nn
from deploy import HERE, PACKAGE, PROMPTS, load_package
from sam3.model.data_misc import FindStage


def register_antialias_resize_symbolic(tensorrt_compatible=False):
    """把固定形状P8抗锯齿双线性插值映射为ONNX Resize-18。"""
    from torch.onnx import register_custom_op_symbolic, symbolic_helper
    def symbolic(g,image,output_size,align_corners,scales_h=None,scales_w=None):
        size=symbolic_helper._maybe_get_const(output_size,'is')
        if not isinstance(size,(list,tuple)) and output_size.node().kind() == 'prim::ListConstruct':
            size=[symbolic_helper._maybe_get_const(v,'i')
                  for v in symbolic_helper._unpack_list(output_size)]
        if not isinstance(size,(list,tuple)) or len(size)!=2:
            raise RuntimeError('固定部署导出要求抗锯齿Resize输出尺寸为常量二元组')
        shape=g.op('Shape',image)
        prefix=g.op('Slice',shape,
            g.op('Constant',value_t=torch.tensor([0],dtype=torch.long)),
            g.op('Constant',value_t=torch.tensor([2],dtype=torch.long)),
            g.op('Constant',value_t=torch.tensor([0],dtype=torch.long)))
        spatial_parts=[]
        for value in size:
            if symbolic_helper._is_value(value):
                scalar=value
            else:
                scalar=g.op('Constant',value_t=torch.tensor(value,dtype=torch.long))
            spatial_parts.append(g.op('Unsqueeze',scalar,
                g.op('Constant',value_t=torch.tensor([0],dtype=torch.long))))
        spatial=g.op('Concat',*spatial_parts,axis_i=0)
        sizes=g.op('Concat',prefix,spatial,axis_i=0)
        empty=g.op('Constant',value_t=torch.tensor([],dtype=torch.float32))
        coordinate='align_corners' if symbolic_helper._maybe_get_const(align_corners,'b') else 'half_pixel'
        return g.op('Resize',image,empty,empty,sizes,mode_s='linear',
            coordinate_transformation_mode_s=coordinate,nearest_mode_s='floor',
            antialias_i=0 if tensorrt_compatible else 1)
    register_custom_op_symbolic('aten::_upsample_bilinear2d_aa',symbolic,18)


class FixedSinglePromptNetwork(nn.Module):
    """输入已归一化图像，输出200个候选的原始预测，不含预处理和后处理。"""
    def __init__(self, model, prompt):
        super().__init__(); self.model=model
        text=model.backbone.forward_text([prompt],device=model.device)
        self.register_buffer('language_features',text['language_features'])
        self.register_buffer('language_mask',text['language_mask'])
        self.register_buffer('language_embeds',text['language_embeds'])
        self.register_buffer('img_ids',torch.zeros(1,dtype=torch.long,device=model.device))
        self.register_buffer('text_ids',torch.zeros(1,dtype=torch.long,device=model.device))
        self.register_buffer('empty_boxes',torch.zeros(0,1,4,device=model.device))
        self.register_buffer('empty_box_mask',torch.zeros(1,0,dtype=torch.bool,device=model.device))

    def forward(self,image):
        backbone=self.model.backbone.forward_image(image)
        backbone.update(language_features=self.language_features,
                        language_mask=self.language_mask,
                        language_embeds=self.language_embeds)
        find=FindStage(img_ids=self.img_ids,text_ids=self.text_ids,input_boxes=None,
            input_boxes_mask=None,input_boxes_label=None,input_points=None,input_points_mask=None)
        # 保持原模型空几何提示的构造和计算，不把图像相关几何编码错误地常量化。
        prompt=self.model._get_dummy_prompt()
        output=self.model.forward_grounding(backbone,find,None,prompt)
        return (output['pred_logits'],output['presence_logit_dec'],
                output['pred_boxes'],output['pred_masks'])


@torch.inference_mode()
def main(a):
    if a.output.exists(): raise FileExistsError(f'避免覆盖已有ONNX：{a.output}')
    if not a.dynamo: register_antialias_resize_symbolic(a.tensorrt_compatible)
    model=load_package(a.package).float().to(a.device).eval()
    wrapper=FixedSinglePromptNetwork(model,a.prompt).eval()
    example=torch.zeros(1,3,1008,1008,device=a.device,dtype=torch.float32)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    torch.onnx.export(wrapper,(example,),str(a.output),opset_version=18,
        input_names=['image'],output_names=['pred_logits','presence_logit_dec','pred_boxes','pred_masks'],
        dynamo=a.dynamo,do_constant_folding=False,external_data=True)
    import onnx
    graph=onnx.load(str(a.output),load_external_data=False)
    report={'format':'p12_fixed_single_prompt_onnx_v1','prompt':a.prompt,
        'input':{'name':'image','shape':[1,3,1008,1008],'dtype':'float32_normalized'},
        'outputs':['pred_logits','presence_logit_dec','pred_boxes','pred_masks'],
        'opset':18,'nodes':len(graph.graph.node),'initializers':len(graph.graph.initializer),
        'tensorrt_compatible_resize':a.tensorrt_compatible,
        'onnx_mib':a.output.stat().st_size/2**20,
        'external_files':[p.name for p in a.output.parent.glob(a.output.name+'*') if p!=a.output]}
    a.output.with_suffix('.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--package',type=Path,default=PACKAGE)
    p.add_argument('--prompt',choices=PROMPTS,default=PROMPTS[0])
    p.add_argument('--output',type=Path,default=HERE/'weights/p12_white_solid_fp32.onnx')
    p.add_argument('--dynamo',action='store_true',help='使用torch.export/Dynamo ONNX导出器')
    p.add_argument('--tensorrt-compatible',action='store_true',
                   help='导出时关闭TensorRT不支持的P8 Resize抗锯齿；必须另做精度校验')
    p.add_argument('--device',default='cuda:0'); main(p.parse_args())
