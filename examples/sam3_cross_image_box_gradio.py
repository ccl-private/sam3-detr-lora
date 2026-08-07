"""Experimental cross-image visual-prompt search with SAM 3 and Gradio.

Upload a reference image and a target image, draw positive/negative exemplar boxes
on the reference image, then find matching instances in the target image.

    .venv/bin/python examples/sam3_cross_image_box_gradio.py --host 0.0.0.0
"""

from __future__ import annotations

import argparse
import sys
import threading
from contextlib import nullcontext
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import gradio as gr

import sam3_image_box_gradio as shared
from sam3.model import box_ops
from sam3.model.data_misc import interpolate
from sam3.model.sam3_image_processor import Sam3Processor


_MODEL = None
_MODEL_LOCK = threading.Lock()
_DEFAULT_CHECKPOINT = REPO_ROOT / "sam3.pt"
_CHECKPOINT_PATH: Optional[str] = (
    str(_DEFAULT_CHECKPOINT) if _DEFAULT_CHECKPOINT.is_file() else None
)
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_model():
    global _MODEL
    with _MODEL_LOCK:
        if _MODEL is None:
            kwargs = {"device": _DEVICE}
            if _CHECKPOINT_PATH:
                kwargs.update(
                    checkpoint_path=_CHECKPOINT_PATH,
                    load_from_HF=False,
                )
            _MODEL = shared.build_sam3_image_model(**kwargs)
    return _MODEL


def upload_reference(image):
    if image is None:
        return None, None, shared._empty_prompts(), "请上传提示图。", None, []
    image = shared._as_rgb(image)
    return (
        image,
        image,
        shared._empty_prompts(),
        "在提示图上绘制一个绿色正样例框，可再添加红色抑制框。",
        None,
        [],
    )


def upload_target(image):
    if image is None:
        return None, None, "请上传目标图。", []
    image = shared._as_rgb(image)
    return image, None, "目标图已更新，可以开始检索。", []


def _normalize_box(box, width, height):
    left, top, right, bottom = box
    return [
        ((left + right) / 2) / width,
        ((top + bottom) / 2) / height,
        (right - left) / width,
        (bottom - top) / height,
    ]


def _build_reference_embedding(model, processor, image, prompts):
    """Pool visual tokens from boxes in the reference image."""
    reference = Image.fromarray(shared._as_rgb(image))
    width, height = reference.size
    reference_state = processor.set_image(reference)
    reference_state["backbone_out"].update(
        model.backbone.forward_text(["visual"], device=_DEVICE)
    )

    geometric_prompt = model._get_dummy_prompt()
    all_boxes = [prompts["positive"], *prompts.get("negative", [])]
    all_labels = [True] + [False] * len(prompts.get("negative", []))
    for box, label in zip(all_boxes, all_labels):
        boxes = torch.tensor(
            _normalize_box(box, width, height),
            device=_DEVICE,
            dtype=torch.float32,
        ).view(1, 1, 4)
        labels = torch.tensor([label], device=_DEVICE, dtype=torch.bool).view(1, 1)
        geometric_prompt.append_boxes(boxes, labels)

    feat_tuple = model._get_img_feats(
        reference_state["backbone_out"], processor.find_stage.img_ids
    )
    _, img_feats, img_pos_embeds, vis_feat_sizes = feat_tuple
    return model.geometry_encoder(
        geo_prompt=geometric_prompt,
        img_feats=img_feats,
        img_sizes=vis_feat_sizes,
        img_pos_embeds=img_pos_embeds,
    )


def _forward_with_visual_embedding(
    model, processor, target_state, visual_prompt_embed, visual_prompt_mask
):
    """Run image grounding with visual tokens extracted from another image."""
    backbone_out = target_state["backbone_out"]
    backbone_out.update(model.backbone.forward_text(["visual"], device=_DEVICE))
    empty_geometry = model._get_dummy_prompt()
    prompt, prompt_mask, backbone_out = model._encode_prompt(
        backbone_out,
        processor.find_stage,
        empty_geometry,
        visual_prompt_embed=visual_prompt_embed,
        visual_prompt_mask=visual_prompt_mask,
    )
    backbone_out, encoder_out, _ = model._run_encoder(
        backbone_out, processor.find_stage, prompt, prompt_mask
    )
    out = {
        "encoder_hidden_states": encoder_out["encoder_hidden_states"],
        "prev_encoder_out": {
            "encoder_out": encoder_out,
            "backbone_out": backbone_out,
        },
    }
    out, hidden_states = model._run_decoder(
        memory=out["encoder_hidden_states"],
        pos_embed=encoder_out["pos_embed"],
        src_mask=encoder_out["padding_mask"],
        out=out,
        prompt=prompt,
        prompt_mask=prompt_mask,
        encoder_out=encoder_out,
    )
    image_ids = processor.find_stage.img_ids
    if "id_mapping" in backbone_out and backbone_out["id_mapping"] is not None:
        image_ids = backbone_out["id_mapping"][image_ids]
    model._run_segmentation_heads(
        out=out,
        backbone_out=backbone_out,
        img_ids=image_ids,
        vis_feat_sizes=encoder_out["vis_feat_sizes"],
        encoder_hidden_states=out["encoder_hidden_states"],
        prompt=prompt,
        prompt_mask=prompt_mask,
        hs=hidden_states,
    )
    return out


def _postprocess(outputs, target_state, threshold):
    probabilities = outputs["pred_logits"].sigmoid()
    presence = outputs["presence_logit_dec"].sigmoid().unsqueeze(1)
    probabilities = (probabilities * presence).squeeze(-1)
    keep = probabilities > float(threshold)
    probabilities = probabilities[keep]
    masks = outputs["pred_masks"][keep]
    boxes = box_ops.box_cxcywh_to_xyxy(outputs["pred_boxes"][keep])

    height = target_state["original_height"]
    width = target_state["original_width"]
    scale = torch.tensor([width, height, width, height], device=_DEVICE)
    boxes = boxes * scale[None, :]
    masks = interpolate(
        masks.unsqueeze(1),
        (height, width),
        mode="bilinear",
        align_corners=False,
    ).sigmoid() > 0.5
    return masks, boxes, probabilities


def run_cross_image(reference, target, prompts, threshold):
    if reference is None:
        raise gr.Error("请先上传提示图。")
    if target is None:
        raise gr.Error("请先上传目标图。")
    prompts = prompts or shared._empty_prompts()
    if prompts.get("positive") is None:
        raise gr.Error("请先在提示图上绘制绿色正样例框。")

    model = get_model()
    processor = Sam3Processor(model, device=_DEVICE, confidence_threshold=threshold)
    autocast = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if _DEVICE == "cuda"
        else nullcontext()
    )
    with _MODEL_LOCK, autocast:
        visual_embed, visual_mask = _build_reference_embedding(
            model, processor, reference, prompts
        )
        target_image = Image.fromarray(shared._as_rgb(target))
        target_state = processor.set_image(target_image)
        outputs = _forward_with_visual_embedding(
            model, processor, target_state, visual_embed, visual_mask
        )
        masks, boxes, scores = _postprocess(outputs, target_state, threshold)

    masks_np = masks.detach().cpu().numpy()
    boxes_np = boxes.detach().float().cpu().numpy()
    scores_np = scores.detach().float().cpu().numpy()
    rendered = shared._render_results(
        target, masks_np, boxes_np, scores_np, prompt_box=None
    )
    rows = [
        [
            index + 1,
            round(float(score), 4),
            round(float(x1), 1),
            round(float(y1), 1),
            round(float(x2), 1),
            round(float(y2), 1),
        ]
        for index, ((x1, y1, x2, y2), score) in enumerate(
            zip(boxes_np, scores_np)
        )
    ]
    return (
        rendered,
        rows,
        f"完成：目标图保留 {len(rows)} 个 Score > {float(threshold):.2f} 的结果。",
    )


def build_demo():
    with gr.Blocks(
        title="SAM 3 跨图视觉提示检索",
        css=shared.CROSSHAIR_CSS,
        js=shared.CROSSHAIR_JS,
    ) as demo:
        gr.Markdown(
            "# SAM 3：在提示图框选，在目标图查找\n"
            "左图绿色框提供正样例，红色框提供负样例；右图输出匹配实例。"
            "这是基于 SAM3 内部 visual prompt embedding 的实验性跨图模式。"
        )
        reference_state = gr.State(None)
        target_state = gr.State(None)
        prompts_state = gr.State(shared._empty_prompts())

        with gr.Row():
            with gr.Column():
                reference_image = gr.Image(
                    label="1. 上传提示图并画框",
                    type="numpy",
                    interactive=True,
                    elem_classes="box-prompt-image",
                )
                box_mode = gr.Radio(
                    ["正样例框", "抑制框"],
                    value="正样例框",
                    label="当前绘制类型",
                )
                with gr.Row():
                    undo_button = gr.Button("撤销最后抑制框")
                    clear_button = gr.Button("清除所有框")
            with gr.Column():
                target_image = gr.Image(
                    label="2. 上传目标图", type="numpy", interactive=True
                )
                threshold = gr.Slider(
                    0.05,
                    0.95,
                    value=0.5,
                    step=0.05,
                    label="Score 过滤阈值",
                )
                run_button = gr.Button("在目标图查找", variant="primary")

        status = gr.Textbox(label="状态", interactive=False)
        with gr.Row():
            result_image = gr.Image(label="目标图检索结果", type="pil")
            results_table = gr.Dataframe(
                headers=["ID", "Score", "x1", "y1", "x2", "y2"],
                datatype=["number"] * 6,
                interactive=False,
            )

        reference_image.upload(
            upload_reference,
            inputs=reference_image,
            outputs=[
                reference_image,
                reference_state,
                prompts_state,
                status,
                result_image,
                results_table,
            ],
        )
        target_image.upload(
            upload_target,
            inputs=target_image,
            outputs=[target_state, result_image, status, results_table],
        )
        reference_image.select(
            shared.select_corner,
            inputs=[reference_state, prompts_state, box_mode],
            outputs=[
                reference_image,
                prompts_state,
                status,
                result_image,
                results_table,
            ],
        )
        clear_button.click(
            shared.clear_boxes,
            inputs=reference_state,
            outputs=[
                reference_image,
                prompts_state,
                status,
                result_image,
                results_table,
            ],
        )
        undo_button.click(
            shared.undo_negative,
            inputs=[reference_state, prompts_state, box_mode],
            outputs=[
                reference_image,
                prompts_state,
                status,
                result_image,
                results_table,
            ],
        )
        run_button.click(
            run_cross_image,
            inputs=[reference_state, target_state, prompts_state, threshold],
            outputs=[result_image, results_table, status],
        )
    return demo


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument("--share", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.checkpoint is not None:
        if not args.checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
        _CHECKPOINT_PATH = str(args.checkpoint.resolve())
    build_demo().queue(default_concurrency_limit=1).launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
    )
