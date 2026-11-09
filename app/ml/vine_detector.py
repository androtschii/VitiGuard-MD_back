"""Стволы и грозди на фото ряда: YOLO11n (ml/notebooks/vine_colab.ipynb) на ONNX Runtime.

Та же модель, что выкатывается на робота (`/ai/models/vine/file`), на сервере
отвечает на фото с телефона: где на снимке стволы лозы и грозди."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from app.ml.classifier import create_session

DEFAULT_CLASSES = ("trunk", "bunch")
# Порог уверенности и перекрытия для отсева дублей — как у Ultralytics по умолчанию
CONFIDENCE = 0.25
IOU = 0.45
MAX_DETECTIONS = 100
# Серый фон полей при вписывании снимка в квадрат, как при обучении
PAD_COLOR = (114, 114, 114)


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    # Рамка в долях ширины и высоты исходного снимка: x1, y1, x2, y2
    box: tuple[float, float, float, float]


@dataclass(frozen=True)
class Letterbox:
    pixels: np.ndarray
    scale: float
    left: int
    top: int


def letterbox(image: Image.Image, size: int) -> Letterbox:
    """Вписывает снимок в квадрат size×size без искажения пропорций."""
    width, height = image.size
    scale = min(size / width, size / height)
    new_w, new_h = max(1, round(width * scale)), max(1, round(height * scale))
    resized = image.convert("RGB").resize((new_w, new_h), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (size, size), PAD_COLOR)
    left, top = (size - new_w) // 2, (size - new_h) // 2
    canvas.paste(resized, (left, top))
    pixels = np.asarray(canvas, dtype=np.float32) / 255
    batch = np.ascontiguousarray(pixels.transpose(2, 0, 1)[None])
    return Letterbox(batch, scale, left, top)


def iou(box: np.ndarray, others: np.ndarray) -> np.ndarray:
    x1 = np.maximum(box[0], others[:, 0])
    y1 = np.maximum(box[1], others[:, 1])
    x2 = np.minimum(box[2], others[:, 2])
    y2 = np.minimum(box[3], others[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = (box[2] - box[0]) * (box[3] - box[1])
    areas = (others[:, 2] - others[:, 0]) * (others[:, 3] - others[:, 1])
    union = area + areas - inter
    result: np.ndarray = np.divide(
        inter, union, out=np.zeros_like(inter), where=union > 0
    )
    return result


def nms(boxes: np.ndarray, scores: np.ndarray, threshold: float) -> list[int]:
    """Из перекрывающихся рамок одного класса остаётся самая уверенная."""
    order = list(np.argsort(-scores))
    keep: list[int] = []
    while order:
        best = order.pop(0)
        keep.append(int(best))
        if not order:
            break
        rest = np.array(order)
        overlap = iou(boxes[best], boxes[rest])
        order = [int(i) for i, o in zip(rest, overlap, strict=True) if o <= threshold]
    return keep


class VineDetector:
    def __init__(self, model_path: Path) -> None:
        meta_path = model_path.with_suffix(".json")
        meta = (
            json.loads(meta_path.read_text(encoding="utf-8"))
            if meta_path.exists()
            else {}
        )
        self.name: str = meta.get("name", "vine_yolo")
        self.classes: list[str] = list(meta.get("classes", DEFAULT_CLASSES))
        self.session = create_session(model_path)
        model_input = self.session.get_inputs()[0]
        self.input_name: str = model_input.name
        # Размер входа зашит в экспортированную сеть
        side = model_input.shape[-1]
        self.image_size: int = (
            side if isinstance(side, int) else meta.get("image_size", 640)
        )

    def detect(
        self, image: Image.Image, confidence: float = CONFIDENCE
    ) -> list[Detection]:
        width, height = image.size
        box = letterbox(image, self.image_size)
        # Выход YOLO: [1, 4 + классы, рамки] — центр, ширина, высота и оценки классов
        output = np.asarray(self.session.run(None, {self.input_name: box.pixels})[0])
        rows = output[0].T
        scores = rows[:, 4:]
        labels = scores.argmax(axis=1)
        best = scores[np.arange(len(rows)), labels]
        mask = best >= confidence
        rows, labels, best = rows[mask], labels[mask], best[mask]
        cx, cy, w, h = rows[:, 0], rows[:, 1], rows[:, 2], rows[:, 3]
        corners = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
        # Из квадрата — обратно в координаты исходного снимка
        corners[:, [0, 2]] = (corners[:, [0, 2]] - box.left) / box.scale
        corners[:, [1, 3]] = (corners[:, [1, 3]] - box.top) / box.scale
        corners[:, [0, 2]] = np.clip(corners[:, [0, 2]], 0, width) / width
        corners[:, [1, 3]] = np.clip(corners[:, [1, 3]], 0, height) / height

        found: list[Detection] = []
        for label in np.unique(labels):
            index = np.flatnonzero(labels == label)
            for kept in nms(corners[index], best[index], IOU):
                i = index[kept]
                x1, y1, x2, y2 = (round(float(v), 4) for v in corners[i])
                found.append(
                    Detection(
                        self.classes[int(label)],
                        round(float(best[i]), 4),
                        (x1, y1, x2, y2),
                    )
                )
        found.sort(key=lambda item: item.confidence, reverse=True)
        return found[:MAX_DETECTIONS]
