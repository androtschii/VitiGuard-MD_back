from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from vine_model import build_vine_model

from app.ml.vine_detector import VineDetector, iou, letterbox, nms


def test_letterbox_keeps_proportions() -> None:
    # Снимок 128×64 вписывается в 32×16 и встаёт по центру с полями 8 px
    box = letterbox(Image.new("RGB", (128, 64), (255, 255, 255)), 32)

    assert box.pixels.shape == (1, 3, 32, 32)
    assert (box.scale, box.left, box.top) == (0.25, 0, 8)
    assert box.pixels[0, 0, 0, 0] == pytest.approx(114 / 255)
    assert box.pixels[0, 0, 16, 16] == pytest.approx(1)


def test_iou_and_nms() -> None:
    boxes = np.array(
        [[0, 0, 10, 10], [1, 0, 11, 10], [20, 20, 30, 30], [5, 5, 5, 5]],
        dtype=np.float32,
    )
    assert iou(boxes[0], boxes[1:2])[0] == pytest.approx(90 / 110)
    # Пустая рамка не делит на ноль
    assert iou(boxes[3], boxes[3:4])[0] == 0

    keep = nms(boxes[:3], np.array([0.5, 0.9, 0.7], dtype=np.float32), 0.45)
    assert keep == [1, 2]


def test_detect_maps_boxes_to_photo(tmp_path: Path) -> None:
    detector = VineDetector(build_vine_model(tmp_path / "vine.onnx"))
    assert (detector.name, detector.image_size) == ("tiny-yolo", 32)

    found = detector.detect(Image.new("RGB", (128, 64)))

    # Дубль ствола и слабая рамка отсеяны
    assert [item.label for item in found] == ["trunk", "bunch"]
    assert [item.confidence for item in found] == pytest.approx([0.9, 0.7])
    # Ствол: x 12..20, y 8..24 в квадрате → весь снимок по высоте
    assert found[0].box == pytest.approx((0.375, 0, 0.625, 1))
    assert found[1].box == pytest.approx((0.1875, 0.125, 0.3125, 0.375))


def test_detect_threshold_and_defaults(tmp_path: Path) -> None:
    detector = VineDetector(build_vine_model(tmp_path / "vine.onnx", with_meta=False))
    assert detector.name == "vine_yolo"
    assert detector.classes == ["trunk", "bunch"]

    found = detector.detect(Image.new("RGB", (64, 64)), confidence=0.05)
    assert [item.label for item in found] == ["trunk", "bunch", "bunch"]
    assert detector.detect(Image.new("RGB", (64, 64)), confidence=0.95) == []
