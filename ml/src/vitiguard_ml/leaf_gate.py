"""Отсев «не лист винограда» по признакам модели диагностики (ml/pr-021).

Модель знает только болезни винограда и на любом снимке выбирает одну из них:
лист бумаги она называла оидиумом (белая поверхность похожа на налёт), а её
уверенность на бумаге была выше, чем на настоящих полевых листьях. Поверх
признаков той же сети (global pool EfficientNet-B0, 1280 чисел) обучается
логистическая регрессия «виноград / не виноград»: переобучать саму модель не
нужно, и отсев учится на обычном процессоре за минуту.

    uv run --with onnx --with onnxruntime python -m vitiguard_ml.leaf_gate \\
        --model ../models/diagnosis.onnx \\
        --grape data/processed/plantvillage data/processed/ngld \\
        --field data/processed/hermos \\
        --not-grape <предметы> <листья других культур>

Команда добавляет к модели выход features и пишет рядом diagnosis.gate.json;
сервер включает отсев, когда видит оба (app/ml/classifier.py)."""

import argparse
import json
import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

FEATURES = "features"
# Выход global pool экспортированной EfficientNet-B0 (vitiguard_ml.export_server)
POOLED = "/global_pool/flatten/Flatten_output_0"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass(frozen=True)
class Gate:
    mean: np.ndarray
    std: np.ndarray
    weights: np.ndarray
    bias: float
    threshold: float = 0.5

    def scores(self, features: np.ndarray) -> np.ndarray:
        z = ((features - self.mean) / self.std) @ self.weights + self.bias
        return np.asarray(1 / (1 + np.exp(-z)))

    def to_json(self, metrics: dict[str, float] | None = None) -> str:
        return json.dumps(
            {
                "mean": self.mean.tolist(),
                "std": self.std.tolist(),
                "weights": self.weights.tolist(),
                "bias": self.bias,
                "threshold": self.threshold,
                "metrics": metrics or {},
            }
        )


def fit_gate(
    grape: np.ndarray,
    not_grape: np.ndarray,
    steps: int = 800,
    learning_rate: float = 0.05,
    l2: float = 1e-3,
) -> Gate:
    """Логистическая регрессия градиентным спуском. Классы уравновешены весами:
    примеров «не винограда» обычно больше, и без весов порог смещался бы."""
    x = np.concatenate([grape, not_grape]).astype(np.float64)
    y = np.concatenate([np.ones(len(grape)), np.zeros(len(not_grape))])
    mean, std = x.mean(0), x.std(0) + 1e-6
    xs = (x - mean) / std
    balance = np.where(y == 1, len(y) / (2 * len(grape)), len(y) / (2 * len(not_grape)))
    weights = np.zeros(xs.shape[1])
    bias = 0.0
    for _ in range(steps):
        p = 1 / (1 + np.exp(-(xs @ weights + bias)))
        gradient = (p - y) * balance
        weights -= learning_rate * (xs.T @ gradient / len(y) + l2 * weights)
        bias -= learning_rate * float(gradient.mean())
    return Gate(mean, std, weights, bias)


def with_threshold(gate: Gate, field: np.ndarray, keep: float = 0.98) -> Gate:
    """Порог по полевым фото винограда: проходит доля keep настоящих листьев.
    По лабораторным снимкам порог был бы строже и в поле отсекал бы листья."""
    threshold = float(np.quantile(gate.scores(field), 1 - keep))
    return Gate(gate.mean, gate.std, gate.weights, gate.bias, threshold)


def rates(gate: Gate, grape: np.ndarray, not_grape: np.ndarray) -> dict[str, float]:
    return {
        "grape_passed": float(np.mean(gate.scores(grape) >= gate.threshold)),
        "not_grape_rejected": float(np.mean(gate.scores(not_grape) < gate.threshold)),
    }


def add_features_output(model_path: Path, out_path: Path, tensor: str = POOLED) -> None:
    """Выход features у уже экспортированной модели: ONNX Runtime отдаёт только
    выходы графа, промежуточный тензор без этого недоступен."""
    import onnx

    model = onnx.load(model_path)
    if all(output.name != FEATURES for output in model.graph.output):
        model.graph.node.append(onnx.helper.make_node("Identity", [tensor], [FEATURES]))
        model.graph.output.append(
            onnx.helper.make_tensor_value_info(FEATURES, onnx.TensorProto.FLOAT, None)
        )
    onnx.save(model, out_path)


def images(folders: Sequence[Path], limit: int, rng: random.Random) -> list[Path]:
    found = sorted(
        path
        for folder in folders
        for path in folder.rglob("*")
        if path.suffix.lower() in IMAGE_SUFFIXES
    )
    rng.shuffle(found)
    return found[:limit]


def extract(model_path: Path, paths: Sequence[Path]) -> np.ndarray:  # pragma: no cover
    import onnxruntime as ort
    from PIL import Image

    meta: dict[str, Any] = json.loads(model_path.with_suffix(".json").read_text())
    size = int(meta["image_size"])
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    rows = []
    for path in paths:
        image = (
            Image.open(path)
            .convert("RGB")
            .resize((size, size), Image.Resampling.BILINEAR)
        )
        pixels = (np.asarray(image, dtype=np.float32) / 255 - mean) / std
        batch = np.ascontiguousarray(pixels.transpose(2, 0, 1)[None])
        rows.append(session.run([FEATURES], {"image": batch})[0][0])
    return np.asarray(rows, dtype=np.float64)


def main() -> None:  # pragma: no cover — работа с настоящей моделью и данными
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--grape", type=Path, nargs="+", required=True)
    parser.add_argument("--field", type=Path, nargs="+", required=True)
    parser.add_argument("--not-grape", type=Path, nargs="+", required=True)
    parser.add_argument("--keep", type=float, default=0.98)
    parser.add_argument("--per-set", type=int, default=1500)
    args = parser.parse_args()

    rng = random.Random(11)
    add_features_output(args.model, args.model)
    field = images(args.field, 2 * args.per_set, rng)
    grape = images(args.grape, args.per_set, rng) + field[: args.per_set]
    not_grape = images(args.not_grape, 2 * args.per_set, rng)
    split = len(not_grape) * 3 // 4
    features = {
        "grape": extract(args.model, grape),
        "field_check": extract(args.model, field[args.per_set :]),
        "not_grape": extract(args.model, not_grape[:split]),
        "not_grape_check": extract(args.model, not_grape[split:]),
    }
    gate = with_threshold(
        fit_gate(features["grape"], features["not_grape"]),
        features["field_check"],
        args.keep,
    )
    metrics = rates(gate, features["field_check"], features["not_grape_check"])
    args.model.with_suffix(".gate.json").write_text(
        gate.to_json(metrics), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=1))


if __name__ == "__main__":  # pragma: no cover
    main()
