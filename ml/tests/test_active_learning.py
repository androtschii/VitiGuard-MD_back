import json
from pathlib import Path

import pytest

from vitiguard_ml.robot.active_learning import (
    BoxPrediction,
    HardSample,
    box_iou,
    calculate_tta_disagreement,
    calculate_uncertainty,
    export_to_label_studio,
    flip_box_horizontal,
    score_sample,
    select_hard_samples,
)


def test_box_iou() -> None:
    # Идентичные рамки: IoU = 1.0
    b1 = BoxPrediction(cls_id=0, cx=0.5, cy=0.5, w=0.2, h=0.4, conf=0.9)
    assert box_iou(b1, b1) == pytest.approx(1.0)

    # Непересекающиеся рамки: IoU = 0.0
    b2 = BoxPrediction(cls_id=0, cx=0.1, cy=0.1, w=0.1, h=0.1, conf=0.9)
    assert box_iou(b1, b2) == pytest.approx(0.0)

    # Частичное пересечение
    b3 = BoxPrediction(cls_id=0, cx=0.6, cy=0.5, w=0.2, h=0.4, conf=0.9)
    iou = box_iou(b1, b3)
    assert 0.2 < iou < 0.5


def test_flip_box_horizontal() -> None:
    box = BoxPrediction(cls_id=1, cx=0.25, cy=0.60, w=0.20, h=0.30, conf=0.85)
    flipped = flip_box_horizontal(box)

    assert flipped.cx == pytest.approx(0.75)
    assert flipped.cy == pytest.approx(0.60)
    assert flipped.w == pytest.approx(0.20)
    assert flipped.h == pytest.approx(0.30)
    assert flipped.cls_id == 1


def test_calculate_tta_disagreement() -> None:
    # Полное совпадение на прямом и зеркальном кадре
    direct = [BoxPrediction(cls_id=0, cx=0.3, cy=0.5, w=0.2, h=0.4, conf=0.9)]
    flipped = [BoxPrediction(cls_id=0, cx=0.7, cy=0.5, w=0.2, h=0.4, conf=0.9)]
    disagreement = calculate_tta_disagreement(direct, flipped)
    assert disagreement == pytest.approx(0.0)

    # Полный пропуск объекта при отражении
    disagreement_miss = calculate_tta_disagreement(direct, [])
    assert disagreement_miss == pytest.approx(1.0)


def test_calculate_uncertainty() -> None:
    # Уверенное предсказание (conf=0.99) -> низкая неуверенность
    sure_boxes = [BoxPrediction(cls_id=0, cx=0.5, cy=0.5, w=0.1, h=0.1, conf=0.99)]
    unc_sure = calculate_uncertainty(sure_boxes)
    assert unc_sure < 0.1

    # Сомнительное предсказание (conf=0.51) -> высокая неуверенность
    doubt_boxes = [BoxPrediction(cls_id=0, cx=0.5, cy=0.5, w=0.1, h=0.1, conf=0.51)]
    unc_doubt = calculate_uncertainty(doubt_boxes)
    assert unc_doubt > 0.9


def test_score_and_select_hard_samples() -> None:
    b_sure = [BoxPrediction(cls_id=0, cx=0.5, cy=0.5, w=0.1, h=0.1, conf=0.95)]
    b_doubt = [BoxPrediction(cls_id=1, cx=0.5, cy=0.5, w=0.1, h=0.1, conf=0.52)]

    sample_easy = score_sample(b_sure, [flip_box_horizontal(b_sure[0])], "f1", "f1.jpg")
    sample_hard = score_sample(b_doubt, [], "f2", "f2.jpg")

    assert sample_hard.combined_score > sample_easy.combined_score

    selected = select_hard_samples([sample_easy, sample_hard], top_k=1)
    assert len(selected) == 1
    assert selected[0].frame_id == "f2"


def test_export_to_label_studio(tmp_path: Path) -> None:
    out_file = tmp_path / "label_studio.json"
    boxes = [BoxPrediction(cls_id=1, cx=0.4, cy=0.6, w=0.2, h=0.3, conf=0.88)]
    samples = [
        HardSample(
            frame_id="frame_001",
            image_file="frame_001.jpg",
            combined_score=0.75,
            uncertainty=0.5,
            tta_disagreement=1.0,
            predictions=boxes,
        )
    ]

    export_to_label_studio(samples, out_file)
    assert out_file.exists()

    data = json.loads(out_file.read_text(encoding="utf-8"))
    assert len(data) == 1
    task = data[0]
    assert task["data"]["frame_id"] == "frame_001"
    assert len(task["predictions"][0]["result"]) == 1
    val = task["predictions"][0]["result"][0]["value"]
    assert val["rectanglelabels"] == ["bunch"]
    assert val["width"] == pytest.approx(20.0)
    assert val["height"] == pytest.approx(30.0)
