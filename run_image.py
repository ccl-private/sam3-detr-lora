import argparse
from pathlib import Path

from PIL import Image
import matplotlib.pyplot as plt
import torch

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

ROOT = Path(__file__).resolve().parent
DEFAULT_CKPT = ROOT / "sam3.pt"
FALLBACK_CKPT = Path("/home/jx/.cache/modelscope/hub/models/facebook/sam3/sam3.pt")
IMAGE_PATH = ROOT / "assets" / "images" / "test_image.jpg"
BPE_PATH = ROOT / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz"
RUNS_DIR = ROOT / "runs"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def resolve_checkpoint() -> Path:
    if DEFAULT_CKPT.exists():
        return DEFAULT_CKPT
    if FALLBACK_CKPT.exists():
        return FALLBACK_CKPT
    raise FileNotFoundError(
        "Checkpoint not found. Copy sam3.pt into the repo root or update the path."
    )


def visualize(image: Image.Image, masks, boxes, scores, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(image)

    if masks is not None and len(masks) > 0:
        for index, item in enumerate(masks):
            mask = item.detach().cpu().numpy()
            if mask.ndim == 3 and mask.shape[0] == 1:
                mask = mask[0]
            masked = mask.astype(float)
            masked[masked == 0] = float("nan")
            ax.imshow(masked, alpha=0.4, cmap="jet", vmin=0, vmax=1)

    if boxes is not None and len(boxes) > 0:
        for index, item in enumerate(boxes):
            x0, y0, x1, y1 = item.detach().cpu().tolist()
            rect = plt.Rectangle(
                (x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="lime", linewidth=2
            )
            ax.add_patch(rect)
            if scores is not None and index < len(scores):
                ax.text(
                    x0,
                    y0,
                    f"{scores[index].item():.3f}",
                    color="lime",
                    fontsize=10,
                    bbox=dict(facecolor="black", alpha=0.4, pad=2, edgecolor="none"),
                )

    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SAM3 text-prompt segmentation on an image.")
    parser.add_argument("--image", type=Path, default=IMAGE_PATH, help="Input image path")
    parser.add_argument("--prompt", default="shoe", help="Text prompt")
    parser.add_argument(
        "--output", type=Path, default=RUNS_DIR / "image_vis.png", help="Output visualization path"
    )
    parser.add_argument("--threshold", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--limit", type=int, help="Process only the first N sorted images")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()

    ckpt_path = resolve_checkpoint()
    model = build_sam3_image_model(
        bpe_path=str(BPE_PATH),
        checkpoint_path=str(ckpt_path),
        device=device,
        eval_mode=True,
    )
    processor = Sam3Processor(model, device=device, confidence_threshold=args.threshold)

    if args.image.is_dir():
        image_paths = sorted(
            path for path in args.image.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES
        )
        if args.limit is not None:
            image_paths = image_paths[: args.limit]
        output_paths = [args.output / f"{path.stem}_vis.png" for path in image_paths]
    else:
        image_paths = [args.image]
        output_paths = [args.output]

    if not image_paths:
        raise FileNotFoundError(f"No supported images found in {args.image}")

    for image_path, out_path in zip(image_paths, output_paths):
        image = Image.open(image_path).convert("RGB")
        state = processor.set_image(image)
        out = processor.set_text_prompt(state=state, prompt=args.prompt)

        masks, boxes, scores = out["masks"], out["boxes"], out["scores"]
        print(f"{image_path.name}: detections={len(scores)}, scores={scores.tolist()}")

        out_path.parent.mkdir(parents=True, exist_ok=True)
        visualize(image, masks, boxes, scores, out_path)
        print("saved:", out_path)


if __name__ == "__main__":
    main()
