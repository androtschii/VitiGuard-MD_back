"""Классификатор болезней по фото листа на ONNX Runtime.

Модель обучена в ml/ (vitiguard_ml.experiments) и экспортирована в ONNX, поэтому
серверу не нужен PyTorch: ONNX Runtime с NumPy в десятки раз легче."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

# Нормализация ImageNet — та же, что при обучении (vitiguard_ml.augment)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def to_input(image: Image.Image, size: int) -> np.ndarray:
    """RGB → квадрат size×size → нормализация → пакет NCHW из одного снимка,
    как в преобразовании оценки при обучении (Resize + Normalize)."""
    resized = image.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
    pixels = np.asarray(resized, dtype=np.float32) / 255
    normalized = (pixels - IMAGENET_MEAN) / IMAGENET_STD
    return np.ascontiguousarray(normalized.transpose(2, 0, 1)[None])


def create_session(model_path: Path) -> ort.InferenceSession:
    options = ort.SessionOptions()
    # Запрос — один снимок: много потоков внутри операции почти не ускоряют
    # его, но отнимают процессор у параллельных запросов
    options.intra_op_num_threads = 2
    return ort.InferenceSession(
        str(model_path), options, providers=["CPUExecutionProvider"]
    )


@dataclass(frozen=True)
class Prediction:
    disease: str
    probability: float


@dataclass(frozen=True)
class Classification:
    # Вероятности всех классов по убыванию
    predictions: list[Prediction]
    # Карта активации самого вероятного класса (0..1) или None, если модель
    # экспортирована без неё
    activation: np.ndarray | None


class LeafClassifier:
    def __init__(self, model_path: Path) -> None:
        meta = json.loads(model_path.with_suffix(".json").read_text(encoding="utf-8"))
        self.name: str = meta["name"]
        self.classes: list[str] = meta["classes"]
        self.image_size: int = meta["image_size"]
        self.session = create_session(model_path)
        self.input_name: str = self.session.get_inputs()[0].name
        # Второй выход — карта активации классов (vitiguard_ml.export_server)
        self.has_activation = len(self.session.get_outputs()) > 1

    def classify(self, image: Image.Image) -> Classification:
        outputs = self.session.run(
            None, {self.input_name: to_input(image, self.image_size)}
        )
        logits = np.asarray(outputs[0][0], dtype=np.float64)
        exp = np.exp(logits - logits.max())
        probabilities = exp / exp.sum()
        best = int(probabilities.argmax())
        ranked = sorted(
            zip(self.classes, probabilities.tolist(), strict=True),
            key=lambda pair: pair[1],
            reverse=True,
        )
        activation = None
        if self.has_activation:
            # Как в Grad-CAM: отрицательный вклад отбрасывается, максимум — 1
            cam = np.maximum(np.asarray(outputs[1][0][best], dtype=np.float32), 0)
            peak = float(cam.max())
            activation = cam / peak if peak > 0 else cam
        return Classification(
            [Prediction(name, probability) for name, probability in ranked],
            activation,
        )
