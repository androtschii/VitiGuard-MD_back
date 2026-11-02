"""Модель LightGBM: заражение милдью по «трём десяткам» на следующий день
(ml/pr-018, подключение pr-076). Обучена на архиве погоды по районам Молдовы,
экспортирована в ONNX; рядом лежит описание с порядком признаков и порогом."""

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

from app.services.risk_features import FEATURES


class RiskModel:
    def __init__(self, path: Path) -> None:
        meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        # Модель с другим порядком признаков молча давала бы неверный ответ
        if tuple(meta["features"]) != FEATURES:
            raise ValueError("Признаки модели риска не совпадают с сервером")
        self.name = str(meta["name"])
        self.threshold = float(meta["threshold"])
        self.session = ort.InferenceSession(
            str(path), providers=["CPUExecutionProvider"]
        )
        self.input = self.session.get_inputs()[0].name

    def probability(self, vector: list[float]) -> float:
        """Вероятность заражения для одного вектора признаков."""
        features = np.asarray([vector], dtype=np.float32)
        outputs = self.session.run(None, {self.input: features})
        probabilities = np.asarray(outputs[-1], dtype=np.float32)
        return float(probabilities[0, 1])
