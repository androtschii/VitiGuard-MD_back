import json
from pathlib import Path
from typing import Any

import pytest
from conftest import make_image

from vitiguard_ml.data.mendeley import (
    parse_via_json,
    polygon_to_bbox,
    prepare_mendeley,
)


def test_polygon_to_bbox_valid() -> None:
    # Полигон 100x200 на картинке 1000x1000: x=[100, 200, 200, 100], y=[200, 200, 400, 400]
    px = [100.0, 200.0, 200.0, 100.0]
    py = [200.0, 200.0, 400.0, 400.0]
    bbox = polygon_to_bbox(px, py, width=1000.0, height=1000.0)

    assert bbox is not None
    cx, cy, w, h = bbox
    assert cx == pytest.approx(0.15)
    assert cy == pytest.approx(0.30)
    assert w == pytest.approx(0.10)
    assert h == pytest.approx(0.20)


def test_polygon_to_bbox_invalid() -> None:
    assert polygon_to_bbox([], [], 100.0, 100.0) is None
    assert polygon_to_bbox([10.0], [10.0, 20.0], 100.0, 100.0) is None
    assert polygon_to_bbox([10.0, 20.0], [10.0, 20.0], 0.0, 100.0) is None


def test_parse_via_json(tmp_path: Path) -> None:
    img_dir = tmp_path / "images"
    img_dir.mkdir(parents=True)
    make_image(img_dir / "vine_001.jpg", seed=1, size=(640, 480))

    via_data = {
        "_via_img_metadata": {
            "vine_001.jpg12345": {
                "filename": "vine_001.jpg",
                "size": 12345,
                "regions": [
                    {
                        "shape_attributes": {
                            "name": "polygon",
                            "all_points_x": [100, 200, 200, 100],
                            "all_points_y": [150, 150, 400, 400],
                        },
                        "region_attributes": {"label": "vine trunk"},
                    },
                    {
                        "shape_attributes": {
                            "name": "polygon",
                            "all_points_x": [50, 80, 80, 50],
                            "all_points_y": [50, 50, 100, 100],
                        },
                        "region_attributes": {"label": "pruned shoot"},  # Не ствол
                    },
                ],
            }
        }
    }
    json_path = tmp_path / "via.json"
    json_path.write_text(json.dumps(via_data), encoding="utf-8")

    parsed = parse_via_json(json_path, img_dir)
    assert "vine_001.jpg" in parsed
    boxes = parsed["vine_001.jpg"]
    assert len(boxes) == 1  # Только ствол


def test_prepare_mendeley(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    images_dir = raw_dir / "mendeley" / "images"
    images_dir.mkdir(parents=True)

    via_dict: dict[str, dict[str, Any]] = {}
    for i in range(5):
        fname = f"sample_{i:02d}.jpg"
        make_image(images_dir / fname, seed=i, size=(300, 400))
        via_dict[fname] = {
            "filename": fname,
            "regions": [
                {
                    "shape_attributes": {
                        "name": "polygon",
                        "all_points_x": [50, 100, 100, 50],
                        "all_points_y": [50, 50, 300, 300],
                    },
                    "region_attributes": {"label": "vine trunk"},
                }
            ],
        }

    (raw_dir / "mendeley" / "via_annotations.json").write_text(
        json.dumps(via_dict), encoding="utf-8"
    )

    out = tmp_path / "processed"
    counts = prepare_mendeley(raw_dir, out, val_ratio=0.20)

    assert counts["test"] == 0
    assert counts["train"] == 4
    assert counts["val"] == 1
    assert (out / "images" / "train" / "mendeley_sample_00.jpg").exists()
    assert (out / "labels" / "train" / "mendeley_sample_00.txt").exists()
    label_content = (out / "labels" / "train" / "mendeley_sample_00.txt").read_text()
    assert label_content.startswith("0 ")
