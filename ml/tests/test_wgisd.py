from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from vitiguard_ml.data.wgisd import (
    convert_detect,
    prepare,
    split_names,
    variety,
)
from vitiguard_ml.robot.bunches import summarize

NAMES = [f"CDY_{index:04d}" for index in range(7)] + ["SYH_2017-04-27_1233"]


def fake_wgisd(root: Path, masked: bool = False) -> Path:
    data = root / "data"
    data.mkdir(parents=True)
    for name in NAMES:
        Image.new("RGB", (40, 30), (90, 30, 120)).save(data / f"{name}.jpg")
        (data / f"{name}.txt").write_text("0 0.5 0.5 0.2 0.3\n")
        if masked:
            masks = np.zeros((30, 40, 2), dtype=bool)
            masks[5:20, 5:15, 0] = True
            masks[0:1, 0:1, 1] = True  # точка: контура из неё нет
            np.savez(data / f"{name}.npz", masks)
    (root / "train.txt").write_text("\n".join(NAMES[:-1]))
    (root / "test.txt").write_text(NAMES[-1])
    (root / "train_masked.txt").write_text("\n".join(NAMES[:-1]))
    (root / "test_masked.txt").write_text(NAMES[-1])
    return root


def test_variety_from_file_name() -> None:
    assert variety("CSV_1878") == "cabernet_sauvignon"
    assert variety("SYH_2017-04-27_1233") == "syrah"
    assert variety("XYZ_1") == "unknown"


def test_split_is_stable_and_takes_every_sixth_per_variety() -> None:
    splits = split_names(NAMES[:-1], NAMES[-1:])

    assert splits["val"] == ["CDY_0005"]
    assert "CDY_0005" not in splits["train"]
    assert splits["test"] == NAMES[-1:]
    assert split_names(list(reversed(NAMES[:-1])), [])["val"] == ["CDY_0005"]


def test_detect_layout_for_ultralytics(tmp_path: Path) -> None:
    root = fake_wgisd(tmp_path / "wgisd")
    splits = split_names(NAMES[:-1], NAMES[-1:])

    count = convert_detect(root / "data", tmp_path / "out", splits)

    assert count == len(NAMES)
    assert (tmp_path / "out/images/val/CDY_0005.jpg").exists()
    assert (tmp_path / "out/labels/test/SYH_2017-04-27_1233.txt").read_text() == (
        "0 0.5 0.5 0.2 0.3\n"
    )
    config = (tmp_path / "out/data.yaml").read_text()
    assert "val: images/val" in config
    assert "0: bunch" in config


def test_masks_become_polygons(tmp_path: Path) -> None:
    pytest.importorskip("cv2")
    root = fake_wgisd(tmp_path / "wgisd", masked=True)

    counts = prepare(root, tmp_path / "out")

    assert counts == {"detect": len(NAMES), "segment": len(NAMES)}
    label = (tmp_path / "out/segment/labels/train/CDY_0000.txt").read_text()
    assert label.count("\n") == 0  # точечная маска отброшена
    values = [float(value) for value in label.split()]
    assert values[0] == 0
    assert len(values[1:]) >= 6
    assert all(0 <= value <= 1 for value in values[1:])


def test_metrics_summary() -> None:
    metrics = SimpleNamespace(
        box=SimpleNamespace(map50=0.9, map=0.6, mp=0.8, mr=0.7),
        seg=SimpleNamespace(map50=0.85, map=0.5),
    )

    assert summarize(metrics) == {
        "box_map50": 0.9,
        "box_map50_95": 0.6,
        "mask_map50": 0.85,
        "mask_map50_95": 0.5,
        "precision": 0.8,
        "recall": 0.7,
    }
