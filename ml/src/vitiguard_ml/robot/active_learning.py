"""Предразметка и активный отбор сложных кадров для робота (ml/pr-030).

Робот собирает сотни кадров в день. Размечать все вручную неэффективно.
Модель прогоняет инференс с проверкой на отражённом кадре (Test-Time Augmentation):
- измеряется неуверенность модели (близость к порогу классификации);
- измеряется расхождение ответов на прямом и зеркальном кадре (TTA disagreement);
- отбираются топ-50..100 самых информативных кадров для быстрой ручной правки
  в Label Studio / CVAT перед ночным дообучением.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from vitiguard_ml.data.vine import CLASSES


@dataclass(frozen=True)
class BoxPrediction:
    """Предсказанная рамка YOLO: нормализованные координаты (0..1) и уверенность."""

    cls_id: int
    cx: float
    cy: float
    w: float
    h: float
    conf: float


@dataclass(frozen=True)
class HardSample:
    """Оценённый кандидат для ручной валидации."""

    frame_id: str
    image_file: str
    combined_score: float
    uncertainty: float
    tta_disagreement: float
    predictions: list[BoxPrediction]


def box_iou(a: BoxPrediction, b: BoxPrediction) -> float:
    """Пересечение по объединению (IoU) двух рамок в формате cx, cy, w, h."""
    ax1, ax2 = a.cx - a.w / 2.0, a.cx + a.w / 2.0
    ay1, ay2 = a.cy - a.h / 2.0, a.cy + a.h / 2.0
    bx1, bx2 = b.cx - b.w / 2.0, b.cx + b.w / 2.0
    by1, by2 = b.cy - b.h / 2.0, b.cy + b.h / 2.0

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h

    area_a = a.w * a.h
    area_b = b.w * b.h
    union = area_a + area_b - intersection

    return float(intersection / union) if union > 0 else 0.0


def flip_box_horizontal(box: BoxPrediction) -> BoxPrediction:
    """Отражает рамку по горизонтали для сверки с результатом TTA."""
    return BoxPrediction(
        cls_id=box.cls_id,
        cx=round(1.0 - box.cx, 6),
        cy=box.cy,
        w=box.w,
        h=box.h,
        conf=box.conf,
    )


def calculate_tta_disagreement(
    direct: list[BoxPrediction],
    flipped: list[BoxPrediction],
    iou_threshold: float = 0.5,
) -> float:
    """Вычисляет степень расхождения предсказаний на прямом и отражённом кадрах.

    0.0 — полное совпадение; 1.0 — модель дала абсолютно разные ответы.
    """
    if not direct and not flipped:
        return 0.0
    if not direct or not flipped:
        return 1.0

    # Отражаем рамки flipped обратно
    unflipped = [flip_box_horizontal(b) for b in flipped]

    matched_direct = set()
    matched_unflipped = set()
    conf_diff_sum = 0.0

    for i, d in enumerate(direct):
        best_iou = 0.0
        best_j = -1
        for j, u in enumerate(unflipped):
            if j in matched_unflipped or d.cls_id != u.cls_id:
                continue
            iou = box_iou(d, u)
            if iou > best_iou:
                best_iou = iou
                best_j = j

        if best_iou >= iou_threshold and best_j >= 0:
            matched_direct.add(i)
            matched_unflipped.add(best_j)
            conf_diff_sum += abs(d.conf - unflipped[best_j].conf)

    total_boxes = max(len(direct), len(flipped))
    unmatched_count = total_boxes - len(matched_direct)
    disagreement = (unmatched_count + 0.5 * conf_diff_sum) / total_boxes
    return float(min(1.0, max(0.0, disagreement)))


def calculate_uncertainty(predictions: list[BoxPrediction]) -> float:
    """Вычисляет неуверенность модели по уверенностям детектированных объектов.

    Максимальна, когда уверенности рамок колеблются в зоне неуверенности (0.35..0.65).
    """
    if not predictions:
        return 0.5  # Пустой кадр: умеренная неопределенность

    # Расстояние уверенности от 0.5 (чем ближе к 0.5, тем выше неуверенность)
    uncertainties = [1.0 - 2.0 * abs(p.conf - 0.5) for p in predictions]
    return float(np.mean(uncertainties)) if uncertainties else 0.0


def score_sample(
    direct: list[BoxPrediction],
    flipped: list[BoxPrediction],
    frame_id: str,
    image_file: str,
    w_uncertainty: float = 0.5,
    w_tta: float = 0.5,
) -> HardSample:
    """Объединяет метрики неуверенности и TTA в единую оценку информативности."""
    unc = calculate_uncertainty(direct)
    tta = calculate_tta_disagreement(direct, flipped)
    combined = float(w_uncertainty * unc + w_tta * tta)

    return HardSample(
        frame_id=frame_id,
        image_file=image_file,
        combined_score=round(combined, 4),
        uncertainty=round(unc, 4),
        tta_disagreement=round(tta, 4),
        predictions=direct,
    )


def select_hard_samples(
    candidates: list[HardSample],
    top_k: int = 50,
) -> list[HardSample]:
    """Ранжирует кандидатов по комбинированному скору и выбирает топ-K."""
    sorted_samples = sorted(candidates, key=lambda s: s.combined_score, reverse=True)
    return sorted_samples[:top_k]


def export_to_label_studio(
    samples: list[HardSample],
    output_json: Path,
    image_url_prefix: str = "/data/local-files/?d=frames/",
) -> None:
    """Экспортирует отобранные кадры с предразметкой в формат задач Label Studio.

    Имена from_name и to_name должны совпадать с тегами проекта:
    <Image name="image" value="$image"/> и
    <RectangleLabels name="label" toName="image"> с метками trunk и bunch,
    иначе Label Studio не покажет предразметку.
    """
    tasks = []

    for sample in samples:
        results = []
        for box in sample.predictions:
            x = (box.cx - box.w / 2.0) * 100.0
            y = (box.cy - box.h / 2.0) * 100.0
            width = box.w * 100.0
            height = box.h * 100.0

            results.append(
                {
                    "from_name": "label",
                    "to_name": "image",
                    "type": "rectanglelabels",
                    "value": {
                        "x": round(x, 2),
                        "y": round(y, 2),
                        "width": round(width, 2),
                        "height": round(height, 2),
                        "rectanglelabels": [CLASSES[box.cls_id]],
                    },
                    "score": round(box.conf, 4),
                }
            )

        task = {
            "data": {
                "image": f"{image_url_prefix}{sample.image_file}",
                "frame_id": sample.frame_id,
                "uncertainty_score": sample.combined_score,
            },
            "predictions": [
                {
                    "model_version": "vitiguard-robot-active-v1",
                    "score": sample.combined_score,
                    "result": results,
                }
            ],
        }
        tasks.append(task)

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(tasks, f, ensure_ascii=False, indent=2)
