import numpy as np
from PIL import Image

from vitiguard_ml.robot.bunch_health import (
    analyze_bunch_pigmentation,
    assess_bunch_health,
    crop_bunch,
    evaluate_frame_bunches,
)


def test_crop_bunch_with_bbox() -> None:
    # Изображение 200x100
    img = Image.new("RGB", (200, 100), color=(100, 150, 80))
    # Bbox: центр (0.5, 0.5), ширина 0.4 (80px), высота 0.6 (60px)
    bbox = (0.5, 0.5, 0.4, 0.6)
    cropped = crop_bunch(img, bbox, margin=0.0)

    assert cropped.size[0] == 80
    assert cropped.size[1] == 60


def test_crop_bunch_without_bbox() -> None:
    img = Image.new("RGB", (100, 100), color=(50, 100, 50))
    cropped = crop_bunch(img, None)
    assert cropped.size == img.size


def test_healthy_bunch_assessment() -> None:
    # Создаём синтетическое изображение здоровых ягод (тёмно-синие/фиолетовые ягоды с пруином)
    arr = np.zeros((100, 100, 3), dtype=np.uint8)
    arr[:, :] = [45, 45, 95]  # Здоровый синий оттенок ягод с восковым налётом
    img = Image.fromarray(arr)

    metrics = analyze_bunch_pigmentation(img)
    assert metrics["rot_ratio"] < 0.15

    assessment = assess_bunch_health(img, rot_threshold=0.15)
    assert assessment.status == "healthy"
    assert assessment.should_harvest is True
    assert assessment.action == "harvest"
    assert assessment.confidence >= 0.70


def test_diseased_bunch_assessment() -> None:
    # Создаём синтетическое изображение гнилых ягод (коричнево-бурый некроз с r > g*1.2, r > b*1.3)
    arr = np.zeros((100, 100, 3), dtype=np.uint8)
    arr[:, :] = [120, 60, 30]  # Некротический буро-коричневый цвет гнили
    img = Image.fromarray(arr)

    metrics = analyze_bunch_pigmentation(img)
    assert metrics["rot_ratio"] > 0.50

    assessment = assess_bunch_health(img, rot_threshold=0.15)
    assert assessment.status == "diseased"
    assert assessment.should_harvest is False
    assert assessment.action == "discard"


def test_evaluate_frame_bunches() -> None:
    img = Image.new("RGB", (300, 300), color=(50, 50, 50))
    boxes = [
        (0.25, 0.25, 0.2, 0.2),
        (0.75, 0.75, 0.3, 0.3),
    ]
    results = evaluate_frame_bunches(img, boxes)
    assert len(results) == 2
    assert all(r.status in ("healthy", "diseased") for r in results)

