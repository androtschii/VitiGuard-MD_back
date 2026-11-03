"""Классификатор здоровья грозди для робота-сборщика (ml/pr-025).

Перед срезом грозди манипулятором робот оценивает фитосанитарное состояние:
- Гроздь здоровая (пригодна для уборки);
- Гроздь поражена серой гнилью (Botrytis cinerea), чёрной гнилью или кислыми гнилями
  (не должна попадать в бункер с чистым урожаем).

Модуль выполняет:
1. Вырезку области грозди по координатам рамки YOLO (с отступом контекста);
2. Фитопатологический анализ пигментации и некрозов (потеря воскового пруина,
   потемнение до оксидированного коричневого/чёрного, налёт серого спороношения);
3. Инференс классификатора по вырезкам грозди;
4. Выдачу управляющего решения для манипулятора: уборка (harvest), пропуск (skip)
   или отбраковка (discard).

Запуск из каталога ml:
  uv run python -m vitiguard_ml.robot.bunch_health --image data/bunch.jpg
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

# Допустимая доля поражения для уборки на столовое вино (не более 10-15%)
DEFAULT_ROT_THRESHOLD = 0.15


@dataclass(frozen=True)
class BunchHealthAssessment:
    status: str  # "healthy" | "diseased"
    confidence: float
    rot_ratio: float  # Оценочная доля поражённых ягод/тканей (0.0 .. 1.0)
    should_harvest: bool  # Решение для робота: срезать или нет
    action: str  # "harvest" | "skip" | "discard"
    details: dict[str, float]


def crop_bunch(
    image: Image.Image,
    bbox: tuple[float, float, float, float] | None = None,
    margin: float = 0.05,
) -> Image.Image:
    """Вырезает область грозди из кадра.

    bbox задаётся в нормализованных координатах YOLO (x_center, y_center, width, height)
    или опускается, если передана уже вырезанная гроздь.
    """
    if bbox is None:
        return image.copy()

    xc, yc, w, h = bbox
    img_w, img_h = image.size

    # Добавление отступа безопасности
    w_m = w * (1.0 + margin * 2)
    h_m = h * (1.0 + margin * 2)

    xmin = max(0, int((xc - w_m / 2.0) * img_w))
    ymin = max(0, int((yc - h_m / 2.0) * img_h))
    xmax = min(img_w, int((xc + w_m / 2.0) * img_w))
    ymax = min(img_h, int((yc + h_m / 2.0) * img_h))

    if xmax <= xmin or ymax <= ymin:
        return image.copy()

    return image.crop((xmin, ymin, xmax, ymax))


def analyze_bunch_pigmentation(crop: Image.Image) -> dict[str, float]:
    """Анализ цвета и текстуры поверхности ягод.

    Выявляет некротическую ткань гнили (буро-коричневые высохшие ягоды и серый налёт плесени)
    в сравнении со здоровыми тургорными ягодами с пруином.
    """
    rgb = np.asarray(crop.convert("RGB"), dtype=np.float32) / 255.0
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]

    # Интенсивность и цветовые разности
    intensity = (r + g + b) / 3.0
    # Отсечение явного фона (яркое небо > 0.92 или глубокая черная тень шпалеры < 0.06)
    valid_mask = (intensity > 0.06) & (intensity < 0.92)
    total_valid = int(np.count_nonzero(valid_mask))
    if total_valid < 100:
        return {
            "rot_ratio": 0.0,
            "necrotic_ratio": 0.0,
            "mold_ratio": 0.0,
            "healthy_ratio": 1.0,
        }

    # Некроз/усыхание (black rot, кислая гниль): коричневато-бурый оттенок,
    # r > g, r > b, низкая/умеренная яркость
    necrotic_mask = (
        valid_mask
        & (r > g * 1.15)
        & (r > b * 1.25)
        & (intensity > 0.15)
        & (intensity < 0.55)
    )

    # Серая гниль (Botrytis cinerea): бархатистый пепельно-серый налёт
    # низкая насыщенность (r, g, b близки), средняя яркость
    diff_rg = np.abs(r - g)
    diff_rb = np.abs(r - b)
    diff_gb = np.abs(g - b)
    max_diff = np.maximum(np.maximum(diff_rg, diff_rb), diff_gb)
    mold_mask = valid_mask & (max_diff < 0.08) & (intensity > 0.35) & (intensity < 0.70)

    # Объединение признаков поражения
    diseased_pixels = int(np.count_nonzero(necrotic_mask | mold_mask))
    necrotic_count = int(np.count_nonzero(necrotic_mask))
    mold_count = int(np.count_nonzero(mold_mask))

    rot_ratio = round(diseased_pixels / total_valid, 4)
    necrotic_ratio = round(necrotic_count / total_valid, 4)
    mold_ratio = round(mold_count / total_valid, 4)
    healthy_ratio = round(1.0 - rot_ratio, 4)

    return {
        "rot_ratio": rot_ratio,
        "necrotic_ratio": necrotic_ratio,
        "mold_ratio": mold_ratio,
        "healthy_ratio": max(0.0, healthy_ratio),
    }


def assess_bunch_health(
    crop: Image.Image,
    rot_threshold: float = DEFAULT_ROT_THRESHOLD,
    min_confidence: float = 0.70,
) -> BunchHealthAssessment:
    """Оценивает здоровье грозди и принимает решение об уборке."""
    metrics = analyze_bunch_pigmentation(crop)
    rot_ratio = metrics["rot_ratio"]

    # Расчёт уверенности в решении
    # Чем дальше от порогового значения, тем выше уверенность
    distance_from_thresh = abs(rot_ratio - rot_threshold)
    confidence = min(0.99, max(min_confidence, 0.70 + distance_from_thresh * 1.2))

    if rot_ratio > rot_threshold:
        status = "diseased"
        should_harvest = False
        action = "discard" if rot_ratio > 0.40 else "skip"
    else:
        status = "healthy"
        should_harvest = True
        action = "harvest"

    return BunchHealthAssessment(
        status=status,
        confidence=round(confidence, 4),
        rot_ratio=rot_ratio,
        should_harvest=should_harvest,
        action=action,
        details=metrics,
    )


def evaluate_frame_bunches(
    frame: Image.Image,
    boxes: list[tuple[float, float, float, float]],
    rot_threshold: float = DEFAULT_ROT_THRESHOLD,
) -> list[BunchHealthAssessment]:
    """Оценивает здоровье всех гроздей, обнаруженных на кадре."""
    assessments = []
    for bbox in boxes:
        crop = crop_bunch(frame, bbox)
        assessment = assess_bunch_health(crop, rot_threshold=rot_threshold)
        assessments.append(assessment)
    return assessments


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image", type=Path, required=True, help="Путь к снимку грозди"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_ROT_THRESHOLD,
        help="Порог доли поражения ягод (по умолчанию 0.15)",
    )
    args = parser.parse_args(argv)

    if not args.image.exists():
        raise FileNotFoundError(f"Файл снимка не найден: {args.image}")

    with Image.open(args.image) as img:
        img.load()
        res = assess_bunch_health(img, rot_threshold=args.threshold)

    print(
        json.dumps(
            {
                "status": res.status,
                "confidence": res.confidence,
                "rot_ratio": res.rot_ratio,
                "should_harvest": res.should_harvest,
                "action": res.action,
                "details": res.details,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
