"""Предразметка и активный отбор сложных кадров для робота (ml/pr-030).

Робот собирает сотни кадров в день. Размечать все вручную неэффективно.
Модель прогоняет инференс с проверкой на отражённом кадре (Test-Time Augmentation):
- измеряется неуверенность модели (близость к порогу классификации);
- измеряется расхождение ответов на прямом и зеркальном кадре (TTA disagreement);
- отбираются топ-50..100 самых информативных кадров для быстрой ручной правки
  в Label Studio / CVAT перед ночным дообучением.
"""

import argparse
import importlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from vitiguard_ml.data.images import IMAGE_SUFFIXES
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


def extract_predictions(boxes_data: Any) -> list[BoxPrediction]:
    """Извлекает список BoxPrediction из объекта результатов детекции Ultralytics Boxes."""
    if boxes_data is None or len(boxes_data) == 0:
        return []

    xywhn = boxes_data.xywhn.cpu().numpy()
    confs = boxes_data.conf.cpu().numpy()
    classes = boxes_data.cls.cpu().numpy()

    return [
        BoxPrediction(
            cls_id=int(cls_id),
            cx=round(float(cx), 6),
            cy=round(float(cy), 6),
            w=round(float(w), 6),
            h=round(float(h), 6),
            conf=round(float(conf), 4),
        )
        for (cx, cy, w, h), conf, cls_id in zip(xywhn, confs, classes, strict=True)
    ]


def evaluate_frame_tta(
    model: Any,
    image_path: Path,
    image_size: int = 640,
    conf_threshold: float = 0.25,
) -> tuple[list[BoxPrediction], list[BoxPrediction]]:
    """Прогоняет модель на прямом и горизонтально отражённом кадре (TTA)."""
    from PIL import Image

    with Image.open(image_path) as im:
        rgb = np.asarray(im.convert("RGB"))

    flipped_rgb = np.ascontiguousarray(np.fliplr(rgb))

    res_direct = model.predict(
        rgb, imgsz=image_size, conf=conf_threshold, verbose=False
    )[0]
    res_flipped = model.predict(
        flipped_rgb, imgsz=image_size, conf=conf_threshold, verbose=False
    )[0]

    return extract_predictions(res_direct.boxes), extract_predictions(res_flipped.boxes)


def process_daily_frames(
    model_path: Path,
    frames_dir: Path,
    out_json: Path,
    top_k: int = 50,
    image_size: int = 640,
    conf_threshold: float = 0.25,
) -> list[HardSample]:
    """Прогоняет модель по кадрам дня, ранжирует по сложности и формирует задачи Label Studio."""
    yolo = importlib.import_module("ultralytics").YOLO
    model = yolo(str(model_path))

    candidates: list[HardSample] = []
    for img_path in sorted(frames_dir.iterdir()):
        if img_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        direct, flipped = evaluate_frame_tta(
            model, img_path, image_size=image_size, conf_threshold=conf_threshold
        )
        candidates.append(
            score_sample(
                direct, flipped, frame_id=img_path.stem, image_file=img_path.name
            )
        )

    selected = select_hard_samples(candidates, top_k=top_k)
    export_to_label_studio(selected, out_json)
    return selected


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", type=Path, required=True, help="Веса обученной модели лозы (.pt)"
    )
    parser.add_argument(
        "--frames", type=Path, required=True, help="Папка со снимками робота за день"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("tasks_label_studio.json"),
        help="Выходной JSON",
    )
    parser.add_argument(
        "--top-k", type=int, default=50, help="Количество сложных кадров для отбора"
    )
    parser.add_argument(
        "--image-size", type=int, default=640, help="Размер кадра для инференса"
    )
    args = parser.parse_args(argv)

    selected = process_daily_frames(
        model_path=args.model,
        frames_dir=args.frames,
        out_json=args.out,
        top_k=args.top_k,
        image_size=args.image_size,
    )
    print(f"Отобрано {len(selected)} сложных кадров, задачи сохранены в {args.out}")


if __name__ == "__main__":
    main()
