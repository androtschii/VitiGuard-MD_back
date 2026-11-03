from pathlib import Path

import pytest
from conftest import make_image

from vitiguard_ml.robot.gallery import (
    BoundingBox,
    box_iou,
    build_weekly_error_gallery,
    match_boxes,
    parse_yolo_labels,
)


def test_box_iou() -> None:
    b1 = BoundingBox(cls_id=0, cx=0.5, cy=0.5, w=0.2, h=0.2)
    b2 = BoundingBox(cls_id=0, cx=0.5, cy=0.5, w=0.2, h=0.2)
    assert box_iou(b1, b2) == pytest.approx(1.0)


def test_match_boxes_tp_fp_fn() -> None:
    gt = [
        BoundingBox(cls_id=0, cx=0.3, cy=0.5, w=0.1, h=0.2),  # ствол
        BoundingBox(cls_id=1, cx=0.7, cy=0.5, w=0.1, h=0.2),  # гроздь
    ]
    pred = [
        BoundingBox(cls_id=0, cx=0.31, cy=0.51, w=0.1, h=0.2),  # верно ствол (TP)
        BoundingBox(cls_id=1, cx=0.1, cy=0.1, w=0.05, h=0.05),  # ложная гроздь (FP)
    ]

    tp, fp, fn = match_boxes(gt, pred, iou_thresh=0.5)

    assert len(tp) == 1
    assert tp[0].cls_id == 0
    assert len(fp) == 1
    assert fp[0].cls_id == 1
    assert len(fn) == 1
    assert fn[0].cls_id == 1  # гроздь на 0.7 была пропущена


def test_parse_yolo_labels(tmp_path: Path) -> None:
    label_file = tmp_path / "sample.txt"
    label_file.write_text("0 0.5 0.5 0.2 0.4 0.88\n1 0.7 0.8 0.1 0.2")

    boxes = parse_yolo_labels(label_file)
    assert len(boxes) == 2
    assert boxes[0].cls_id == 0
    assert boxes[0].conf == pytest.approx(0.88)
    assert boxes[1].cls_id == 1
    assert boxes[1].conf == pytest.approx(1.0)


def test_build_weekly_error_gallery(tmp_path: Path) -> None:
    img_dir = tmp_path / "images"
    gt_dir = tmp_path / "gt"
    pred_dir = tmp_path / "pred"
    img_dir.mkdir()
    gt_dir.mkdir()
    pred_dir.mkdir()

    # Создаём снимок с ошибкой детекции
    make_image(img_dir / "frame_01.jpg", seed=5)
    (gt_dir / "frame_01.txt").write_text("0 0.5 0.5 0.2 0.3")
    (pred_dir / "frame_01.txt").write_text("1 0.2 0.2 0.1 0.1")  # FP гроздь, FN ствол

    out_html = tmp_path / "gallery.html"
    summary = build_weekly_error_gallery(img_dir, gt_dir, pred_dir, out_html)

    assert summary["error_frames"] == 1
    assert summary["total_fp"] == 1
    assert summary["total_fn"] == 1
    assert out_html.exists()
    content = out_html.read_text(encoding="utf-8")
    assert "Галерея ошибок" in content
    assert "frame_01.jpg" in content
