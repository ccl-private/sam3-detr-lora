"""Gradio demo: box one exemplar and segment all similar objects in an image.

Run from the repository root:

    python3 -m pip install "gradio>=4.44,<6"
    python3 examples/sam3_image_box_gradio.py

Select a box by clicking its two opposite corners on the uploaded image.
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
from contextlib import nullcontext
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

# Allow running this file directly without installing the local sam3 package.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Some shared machines have a /tmp/gradio directory owned by another user.
# Keep Gradio uploads in a repository-local writable directory by default.
GRADIO_TEMP_DIR = Path(
    os.environ.setdefault("GRADIO_TEMP_DIR", str(REPO_ROOT / ".gradio_tmp"))
)
GRADIO_TEMP_DIR.mkdir(parents=True, exist_ok=True)

import gradio as gr
import numpy as np
import torch
from PIL import Image, ImageDraw

from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


Point = Tuple[int, int]
Box = Tuple[int, int, int, int]

_MODEL = None
_MODEL_LOCK = threading.Lock()
_DEFAULT_CHECKPOINT = REPO_ROOT / "sam3.pt"
_CHECKPOINT_PATH: Optional[str] = (
    str(_DEFAULT_CHECKPOINT) if _DEFAULT_CHECKPOINT.is_file() else None
)
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CROSSHAIR_CSS = """
.box-prompt-image img {
    cursor: crosshair !important;
}
.sam3-crosshair-line {
    position: fixed;
    display: none;
    pointer-events: none;
    z-index: 9999;
    background: rgba(0, 255, 255, 0.95);
    box-shadow: 0 0 1px #000, 0 0 3px #000;
}
.sam3-crosshair-horizontal { height: 1px; }
.sam3-crosshair-vertical { width: 1px; }
"""

CROSSHAIR_JS = r"""
() => {
    const attached = new WeakSet();
    const horizontal = document.createElement("div");
    const vertical = document.createElement("div");
    horizontal.className = "sam3-crosshair-line sam3-crosshair-horizontal";
    vertical.className = "sam3-crosshair-line sam3-crosshair-vertical";
    document.body.append(horizontal, vertical);

    const hide = () => {
        horizontal.style.display = "none";
        vertical.style.display = "none";
    };
    const attach = () => {
        document.querySelectorAll(".box-prompt-image img").forEach((image) => {
            if (attached.has(image)) return;
            attached.add(image);
            image.addEventListener("mousemove", (event) => {
                const rect = image.getBoundingClientRect();
                if (!rect.width || !rect.height) return hide();
                const x = Math.min(Math.max(event.clientX, rect.left), rect.right);
                const y = Math.min(Math.max(event.clientY, rect.top), rect.bottom);
                horizontal.style.left = `${rect.left}px`;
                horizontal.style.top = `${y}px`;
                horizontal.style.width = `${rect.width}px`;
                vertical.style.left = `${x}px`;
                vertical.style.top = `${rect.top}px`;
                vertical.style.height = `${rect.height}px`;
                horizontal.style.display = "block";
                vertical.style.display = "block";
            });
            image.addEventListener("mouseleave", hide);
        });
    };
    attach();
    new MutationObserver(attach).observe(document.body, {
        childList: true,
        subtree: true,
    });
    window.addEventListener("blur", hide);
}
"""


def get_model():
    """Load the model once, on the first inference request."""
    global _MODEL
    with _MODEL_LOCK:
        if _MODEL is None:
            kwargs = {"device": _DEVICE}
            if _CHECKPOINT_PATH:
                kwargs.update(
                    checkpoint_path=_CHECKPOINT_PATH,
                    load_from_HF=False,
                )
            _MODEL = build_sam3_image_model(**kwargs)
    return _MODEL


def _as_rgb(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=-1)
    if array.shape[-1] == 4:
        array = array[..., :3]
    return np.ascontiguousarray(array.astype(np.uint8))


def _box_from_points(points: Sequence[Point]) -> Optional[Box]:
    if len(points) != 2:
        return None
    (x1, y1), (x2, y2) = points
    left, right = sorted((int(x1), int(x2)))
    top, bottom = sorted((int(y1), int(y2)))
    if right - left < 2 or bottom - top < 2:
        return None
    return left, top, right, bottom


def _empty_prompts():
    return {"positive": None, "negative": [], "pending": []}


def _draw_prompts(image: np.ndarray, prompts, mode: str = "正样例框") -> np.ndarray:
    canvas = Image.fromarray(_as_rgb(image)).copy()
    draw = ImageDraw.Draw(canvas)
    radius = max(4, round(min(canvas.size) / 150))
    width = max(2, round(min(canvas.size) / 250))

    positive = prompts.get("positive")
    if positive is not None:
        draw.rectangle(tuple(positive), outline=(0, 255, 80), width=width)
    for box in prompts.get("negative", []):
        draw.rectangle(tuple(box), outline=(255, 50, 50), width=width)

    pending_color = (255, 50, 50) if mode == "抑制框" else (0, 255, 80)
    for x, y in prompts.get("pending", []):
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill=pending_color,
            outline=(0, 0, 0),
            width=1,
        )
    return np.asarray(canvas)


def upload_image(image: Optional[np.ndarray]):
    if image is None:
        return None, None, _empty_prompts(), "请先上传一张图片。", None, []
    image = _as_rgb(image)
    return (
        image,
        image,
        _empty_prompts(),
        "请选择“正样例框”，然后点击目标的两个对角点。",
        None,
        [],
    )


def select_corner(
    original: Optional[np.ndarray], prompts, mode: str, evt: gr.SelectData
):
    if original is None:
        return None, _empty_prompts(), "请先上传图片。", None, []

    prompts = dict(prompts or _empty_prompts())
    prompts["negative"] = list(prompts.get("negative", []))
    points = list(prompts.get("pending", []))
    if len(points) >= 2:
        points.clear()
    x, y = map(int, evt.index)
    height, width = original.shape[:2]
    points.append((min(max(x, 0), width - 1), min(max(y, 0), height - 1)))
    prompts["pending"] = points

    box = _box_from_points(points)
    if len(points) == 1:
        message = f"{mode}第一个角点：({x}, {y})。请点击对角点。"
    elif box is None:
        prompts["pending"] = []
        message = "框太小，请重新点击两个对角点。"
    else:
        left, top, right, bottom = box
        if mode == "抑制框":
            prompts["negative"].append(list(box))
            message = (
                f"已添加抑制框 #{len(prompts['negative'])}："
                f"x={left}, y={top}, w={right-left}, h={bottom-top}。"
            )
        else:
            prompts["positive"] = list(box)
            message = (
                f"正样例框：x={left}, y={top}, w={right-left}, h={bottom-top}。"
            )
        prompts["pending"] = []
    return _draw_prompts(original, prompts, mode), prompts, message, None, []


def clear_boxes(original: Optional[np.ndarray]):
    if original is None:
        return None, _empty_prompts(), "请先上传图片。", None, []
    return original, _empty_prompts(), "所有框已清除。", None, []


def undo_negative(original: Optional[np.ndarray], prompts, mode: str):
    if original is None:
        return None, _empty_prompts(), "请先上传图片。", None, []
    prompts = dict(prompts or _empty_prompts())
    negatives = list(prompts.get("negative", []))
    if negatives:
        negatives.pop()
        message = f"已撤销最后一个抑制框，当前剩余 {len(negatives)} 个。"
    else:
        message = "当前没有抑制框。"
    prompts["negative"] = negatives
    prompts["pending"] = []
    return _draw_prompts(original, prompts, mode), prompts, message, None, []


def _render_results(
    image: np.ndarray,
    masks: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
    prompt_box: Optional[Box] = None,
    negative_boxes: Sequence[Box] = (),
) -> Image.Image:
    base = _as_rgb(image).astype(np.float32)
    colors = np.asarray(
        [
            (255, 70, 70),
            (40, 190, 255),
            (255, 190, 30),
            (180, 90, 255),
            (30, 220, 130),
            (255, 80, 190),
        ],
        dtype=np.float32,
    )
    alpha = 0.48
    for index, mask in enumerate(masks):
        mask = np.squeeze(mask).astype(bool)
        color = colors[index % len(colors)]
        base[mask] = base[mask] * (1.0 - alpha) + color * alpha

    output = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(output)
    image_width, image_height = output.size
    line_width = max(2, round(min(output.size) / 250))
    for index, (box, score) in enumerate(zip(boxes, scores)):
        x1, y1, x2, y2 = (int(round(value)) for value in box)
        x1 = min(max(x1, 0), image_width - 1)
        y1 = min(max(y1, 0), image_height - 1)
        x2 = min(max(x2, 0), image_width - 1)
        y2 = min(max(y2, 0), image_height - 1)
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        color = tuple(colors[index % len(colors)].astype(int).tolist())
        draw.rectangle((x1, y1, x2, y2), outline=color, width=line_width)
        label = f"#{index + 1} {float(score):.3f}"
        text_box = draw.textbbox((0, 0), label)
        label_width = text_box[2] - text_box[0] + 6
        label_height = text_box[3] - text_box[1] + 4
        label_left = min(x1, max(0, image_width - label_width))
        # Prefer above the box; move inside it when it touches the top edge.
        label_top = y1 - label_height if y1 >= label_height else y1
        label_top = min(max(label_top, 0), max(0, image_height - label_height))
        draw.rectangle(
            (
                label_left,
                label_top,
                min(image_width - 1, label_left + label_width),
                min(image_height - 1, label_top + label_height),
            ),
            fill=color,
        )
        draw.text((label_left + 3, label_top + 1), label, fill=(0, 0, 0))

    # Keep positive and negative exemplars visible.
    if prompt_box is not None:
        draw.rectangle(prompt_box, outline=(0, 255, 80), width=line_width)
    for negative_box in negative_boxes:
        draw.rectangle(tuple(negative_box), outline=(255, 50, 50), width=line_width)
    return output


def run_inference(
    original: Optional[np.ndarray], prompts, threshold: float
):
    if original is None:
        raise gr.Error("请先上传图片。")
    prompts = prompts or _empty_prompts()
    if prompts.get("positive") is None:
        raise gr.Error("请先选择“正样例框”，并点击目标的两个对角点。")
    box = tuple(prompts["positive"])
    negative_boxes = [tuple(item) for item in prompts.get("negative", [])]

    image = Image.fromarray(_as_rgb(original))
    width, height = image.size

    def normalize(box_to_normalize: Box):
        left, top, right, bottom = box_to_normalize
        return [
            ((left + right) / 2) / width,
            ((top + bottom) / 2) / height,
            (right - left) / width,
            (bottom - top) / height,
        ]

    processor = Sam3Processor(
        get_model(), device=_DEVICE, confidence_threshold=float(threshold)
    )
    autocast = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if _DEVICE == "cuda"
        else nullcontext()
    )
    with _MODEL_LOCK, autocast:
        state = processor.set_image(image)
        state = processor.add_geometric_prompt(
            box=normalize(box),
            label=True,
            state=state,
        )
        for negative_box in negative_boxes:
            state = processor.add_geometric_prompt(
                box=normalize(negative_box),
                label=False,
                state=state,
            )

    masks = state["masks"].detach().cpu().numpy()
    boxes = state["boxes"].detach().cpu().numpy()
    scores = state["scores"].detach().float().cpu().numpy()
    result = _render_results(
        original, masks, boxes, scores, box, negative_boxes
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
        for index, ((x1, y1, x2, y2), score) in enumerate(zip(boxes, scores))
    ]
    return (
        result,
        rows,
        f"完成：保留 {len(rows)} 个 Score > {float(threshold):.2f} 的目标，"
        f"使用了 {len(negative_boxes)} 个抑制框。",
    )


def build_demo() -> gr.Blocks:
    with gr.Blocks(
        title="SAM 3 视觉样例检索", css=CROSSHAIR_CSS, js=CROSSHAIR_JS
    ) as demo:
        gr.Markdown(
            "# SAM 3：框选一个样例，分割所有相似目标\n"
            "绿色框是要寻找的正样例，红色框是需要排除的负样例。"
            "每个框均通过依次点击两个对角点完成。"
        )
        original_state = gr.State(None)
        prompts_state = gr.State(_empty_prompts())

        with gr.Row():
            with gr.Column():
                input_image = gr.Image(
                    label="上传并框选样例",
                    type="numpy",
                    interactive=True,
                    elem_classes="box-prompt-image",
                )
                box_mode = gr.Radio(
                    choices=["正样例框", "抑制框"],
                    value="正样例框",
                    label="当前绘制类型",
                )
                with gr.Row():
                    undo_negative_button = gr.Button("撤销最后抑制框")
                    clear_button = gr.Button("清除所有框")
                with gr.Row():
                    run_button = gr.Button("查找所有相似目标", variant="primary")
                threshold = gr.Slider(
                    minimum=0.05,
                    maximum=0.95,
                    value=0.5,
                    step=0.05,
                    label="Score 过滤阈值（仅保留高于该分数的目标）",
                )
                status = gr.Textbox(label="状态", interactive=False)
            with gr.Column():
                result_image = gr.Image(label="分割结果", type="pil")
                results_table = gr.Dataframe(
                    headers=["ID", "Score", "x1", "y1", "x2", "y2"],
                    datatype=["number"] * 6,
                    label="检测结果",
                    interactive=False,
                )

        input_image.upload(
            upload_image,
            inputs=input_image,
            outputs=[
                input_image,
                original_state,
                prompts_state,
                status,
                result_image,
                results_table,
            ],
        )
        input_image.select(
            select_corner,
            inputs=[original_state, prompts_state, box_mode],
            outputs=[
                input_image,
                prompts_state,
                status,
                result_image,
                results_table,
            ],
        )
        clear_button.click(
            clear_boxes,
            inputs=original_state,
            outputs=[
                input_image,
                prompts_state,
                status,
                result_image,
                results_table,
            ],
        )
        undo_negative_button.click(
            undo_negative,
            inputs=[original_state, prompts_state, box_mode],
            outputs=[
                input_image,
                prompts_state,
                status,
                result_image,
                results_table,
            ],
        )
        run_button.click(
            run_inference,
            inputs=[original_state, prompts_state, threshold],
            outputs=[result_image, results_table, status],
        )
    return demo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
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
