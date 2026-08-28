"""P10验证闭环正提示频率控制。"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path

CLASS_NAMES = {
    0: "white solid lane line", 1: "yellow solid lane line",
    2: "white dashed lane line", 3: "yellow dashed lane line",
    4: "zebra crossing", 5: "lane barrier", 6: "road teeth marking",
}


def prompt_slug(class_id: int) -> str:
    return CLASS_NAMES[class_id].replace(" ", "_")


def deterministic_keep(
    image_path: Path, class_id: int, epoch: int, rate: float, seed: int,
) -> bool:
    """以图片、类别和epoch生成可复现的伯努利选择。"""
    if rate >= 1.0:
        return True
    if rate <= 0.0:
        return False
    identity = f"{seed}:{epoch}:{image_path.resolve()}:{class_id}".encode("utf-8")
    value = int.from_bytes(hashlib.sha1(identity).digest()[:8], "big") / 2**64
    return value < rate


@dataclass
class ControllerConfig:
    minimum_rate: float = 0.4
    maximum_rate: float = 1.0
    ema_new_weight: float = 0.3
    update_weight: float = 0.2
    maximum_epoch_change: float = 0.1
    deadband: float = 0.02
    performance_gain: float = 2.0
    iou_weight: float = 0.7
    recall_weight: float = 0.3
    minimum_positive_images: int = 20


class AdaptivePromptController:
    """用验证集相对表现平滑调整下一轮各类正提示率。"""

    def __init__(self, base_rates: dict[int, float], config: ControllerConfig) -> None:
        self.config = config
        self.base_rates = {
            class_id: min(config.maximum_rate, max(config.minimum_rate, float(rate)))
            for class_id, rate in base_rates.items()
        }
        self.rates = {class_id: 1.0 for class_id in CLASS_NAMES}
        self.ema_iou: dict[int, float] = {}
        self.ema_recall: dict[int, float] = {}
        self.history: list[dict] = []

    def update(
        self, epoch: int, iou: dict[int, float], recall: dict[int, float],
        positive_images: dict[int, int],
    ) -> dict[int, float]:
        cfg = self.config
        eligible = [
            class_id for class_id in CLASS_NAMES
            if positive_images.get(class_id, 0) >= cfg.minimum_positive_images
        ]
        for class_id in eligible:
            if class_id not in self.ema_iou:
                self.ema_iou[class_id] = float(iou[class_id])
                self.ema_recall[class_id] = float(recall[class_id])
            else:
                self.ema_iou[class_id] = (
                    (1 - cfg.ema_new_weight) * self.ema_iou[class_id]
                    + cfg.ema_new_weight * float(iou[class_id])
                )
                self.ema_recall[class_id] = (
                    (1 - cfg.ema_new_weight) * self.ema_recall[class_id]
                    + cfg.ema_new_weight * float(recall[class_id])
                )
        quality = {
            class_id: cfg.iou_weight * self.ema_iou[class_id]
            + cfg.recall_weight * self.ema_recall[class_id]
            for class_id in eligible
        }
        ordered = sorted(quality.values())
        reference = ordered[len(ordered) // 2] if ordered else 0.0
        previous, targets = dict(self.rates), dict(self.rates)
        for class_id in CLASS_NAMES:
            if class_id not in quality:
                targets[class_id] = self.rates[class_id] = 1.0
                continue
            difference = reference - quality[class_id]
            if abs(difference) < cfg.deadband:
                difference = 0.0
            target = self.base_rates[class_id] * math.exp(
                cfg.performance_gain * difference
            )
            target = min(cfg.maximum_rate, max(cfg.minimum_rate, target))
            targets[class_id] = target
            proposed = previous[class_id] + cfg.update_weight * (
                target - previous[class_id]
            )
            change = min(
                cfg.maximum_epoch_change,
                max(-cfg.maximum_epoch_change, proposed - previous[class_id]),
            )
            self.rates[class_id] = min(
                cfg.maximum_rate,
                max(cfg.minimum_rate, previous[class_id] + change),
            )
        self.history.append({
            "epoch": int(epoch),
            "iou": {str(k): float(v) for k, v in iou.items()},
            "recall": {str(k): float(v) for k, v in recall.items()},
            "positive_images": {str(k): int(v) for k, v in positive_images.items()},
            "ema_iou": {str(k): float(v) for k, v in self.ema_iou.items()},
            "ema_recall": {str(k): float(v) for k, v in self.ema_recall.items()},
            "quality_reference": float(reference),
            "target_rates": {str(k): float(v) for k, v in targets.items()},
            "rates_used": {str(k): float(v) for k, v in previous.items()},
            "rates_next": {str(k): float(v) for k, v in self.rates.items()},
        })
        return dict(self.rates)
