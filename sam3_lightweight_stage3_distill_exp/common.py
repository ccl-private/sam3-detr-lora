from __future__ import annotations

import hashlib
import random
from pathlib import Path

import torch


def cache_file(cache_root: Path, split: str, image_path: Path) -> Path:
    identity = str(image_path.expanduser().resolve()).encode("utf-8")
    name = hashlib.sha1(identity).hexdigest()
    return cache_root / split / f"{name}.pt"


def load_cache(cache_root: Path, split: str, image_path: Path) -> dict:
    path = cache_file(cache_root, split, image_path)
    if not path.exists():
        raise FileNotFoundError(f"缺少教师缓存：{path}，图片={image_path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    expected = str(image_path.expanduser().resolve())
    if payload.get("image_path") != expected:
        raise RuntimeError(f"教师缓存图片不匹配：{path}")
    return payload


def prompt_key(text: str) -> str:
    return " ".join(text.strip().lower().split())


def deterministic_negatives(image_path: Path, candidates: list[str], count: int) -> list[str]:
    cleaned = sorted({prompt_key(text) for text in candidates if prompt_key(text)})
    if count <= 0 or not cleaned:
        return []
    seed = int.from_bytes(
        hashlib.sha1(str(image_path.expanduser().resolve()).encode("utf-8")).digest()[:8],
        byteorder="big",
    )
    return random.Random(seed).sample(cleaned, min(count, len(cleaned)))
