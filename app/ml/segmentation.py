"""Маска поражения листа: U-Net (ml/pr-013) на ONNX Runtime."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from app.ml.classifier import create_session, to_input

# Лист — насыщенные по цвету пиксели, как в правилах разметки масок при обучении
# (vitiguard_ml.masks); фон снимка листа обычно сероватый или белый
MIN_LEAF_SATURATION = 40


@dataclass(frozen=True)
class LesionMask:
    # Маска поражения в размере входа модели
    lesion: np.ndarray
    # Доля поражённой площади листа, 0..1
    severity: float


class LesionSegmenter:
    def __init__(self, model_path: Path) -> None:
        meta = json.loads(model_path.with_suffix(".json").read_text(encoding="utf-8"))
        self.name: str = meta["name"]
        self.image_size: int = meta["image_size"]
        self.session = create_session(model_path)
        self.input_name: str = self.session.get_inputs()[0].name

    def segment(self, image: Image.Image) -> LesionMask:
        size = self.image_size
        logits = self.session.run(None, {self.input_name: to_input(image, size)})[0]
        lesion = np.asarray(logits[0, 0]) > 0  # сигмоида > 0,5
        # Площадь листа приближённо: насыщенные пиксели и всё поражение (белый
        # налёт оидиума ненасыщен, но это лист). Без поиска связных областей:
        # серверу не нужен OpenCV ради одной операции
        hsv = image.convert("RGB").resize((size, size)).convert("HSV")
        saturated = np.asarray(hsv, dtype=np.uint8)[..., 1] >= MIN_LEAF_SATURATION
        leaf = saturated | lesion
        leaf_pixels = int(leaf.sum())
        severity = float(lesion.sum() / leaf_pixels) if leaf_pixels else 0.0
        return LesionMask(lesion, min(severity, 1.0))
