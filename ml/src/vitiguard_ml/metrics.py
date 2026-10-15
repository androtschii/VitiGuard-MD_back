"""Метрики классификации и сегментации на numpy — без scikit-learn и torch,
поэтому их проверяют те же быстрые тесты, что и подготовку данных."""

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray


def confusion_matrix(
    y_true: Sequence[int], y_pred: Sequence[int], num_classes: int
) -> NDArray[np.int64]:
    """Строки — истинный класс, столбцы — предсказанный."""
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for truth, prediction in zip(y_true, y_pred, strict=True):
        matrix[truth, prediction] += 1
    return matrix


def classification_report(
    y_true: Sequence[int], y_pred: Sequence[int], classes: Sequence[str]
) -> dict[str, Any]:
    """Accuracy, precision, recall и F1 по каждому классу и macro-F1.

    Классы несбалансированы, поэтому главная метрика — macro-F1: среднее F1 по
    классам, где редкий оидиум весит столько же, сколько частая эска.
    Классы без примеров в выборке в macro-F1 не входят."""
    matrix = confusion_matrix(y_true, y_pred, len(classes))
    per_class: dict[str, dict[str, float]] = {}
    f1_present: list[float] = []
    for index, name in enumerate(classes):
        true_positive = float(matrix[index, index])
        predicted = float(matrix[:, index].sum())
        support = float(matrix[index, :].sum())
        precision = true_positive / predicted if predicted else 0.0
        recall = true_positive / support if support else 0.0
        f1 = (
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        )
        per_class[name] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": int(support),
        }
        if support:
            f1_present.append(f1)
    total = int(matrix.sum())
    return {
        "accuracy": round(float(np.trace(matrix)) / total, 4) if total else 0.0,
        "macro_f1": round(float(np.mean(f1_present)), 4) if f1_present else 0.0,
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
        "classes": list(classes),
    }


def iou(prediction: NDArray[np.bool_], target: NDArray[np.bool_]) -> float:
    """Пересечение к объединению. Обе маски пустые — полное совпадение."""
    union = np.logical_or(prediction, target).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(prediction, target).sum() / union)


def dice(prediction: NDArray[np.bool_], target: NDArray[np.bool_]) -> float:
    total = prediction.sum() + target.sum()
    if total == 0:
        return 1.0
    return float(2 * np.logical_and(prediction, target).sum() / total)
