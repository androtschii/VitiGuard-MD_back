import json
import random
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from vitiguard_ml.gate_field import (
    frame_key,
    load_gate,
    named_folders,
    paper_images,
    pass_rate,
    report,
    shrink_frames,
    split_by_frame,
    stack,
)
from vitiguard_ml.leaf_gate import Gate


def test_shrink_frames_two_sizes(tmp_path: Path) -> None:
    source = tmp_path / "src"
    source.mkdir()
    Image.new("RGB", (2000, 1000), (0, 120, 0)).save(source / "a.jpg")
    (source / "broken.jpg").write_bytes(b"not an image")

    written = shrink_frames([source], tmp_path / "out", 10, random.Random(1))

    sizes = sorted(Image.open(path).size for path in written)
    assert sizes == [(320, 160), (1024, 512)]
    assert len({frame_key(path) for path in written}) == 1


def test_split_keeps_sizes_of_one_frame_together() -> None:
    paths = [Path(f"{i:05d}_{side}.jpg") for i in range(10) for side in (1024, 320)]

    train, check = split_by_frame(paths, 0.3, random.Random(3))

    assert len(check) == 6
    assert len(train) == 14
    assert not {frame_key(p) for p in train} & {frame_key(p) for p in check}


def test_paper_images_light(tmp_path: Path) -> None:
    written = paper_images(tmp_path, 3, random.Random(5))

    assert len(written) == 3
    for path in written:
        pixels = np.asarray(Image.open(path))
        assert pixels.shape == (320, 320, 3)
        assert pixels.mean() > 120


def test_named_folders() -> None:
    named = named_folders(["a=x", "b=y", "a=z"])
    assert named == {"a": [Path("x"), Path("z")], "b": [Path("y")]}
    with pytest.raises(ValueError, match="имя=папка"):
        named_folders(["no-folder"])


def gate(threshold: float = 0.5) -> Gate:
    return Gate(np.zeros(1), np.ones(1), np.array([4.0]), 0.0, threshold)


def test_pass_rate_and_report() -> None:
    grape = np.array([[1.0], [2.0], [-1.0], [3.0]])
    other = np.array([[-2.0], [-3.0], [1.0], [-1.0]])

    assert pass_rate(gate(), grape) == 0.75
    assert np.isnan(pass_rate(gate(), np.zeros((0, 1))))

    metrics = report({"new": gate()}, {"field": grape}, {"paper": other})
    assert metrics == {
        "new": {"pass_field": 0.75, "reject_paper": 0.75, "threshold": 0.5}
    }


def test_load_gate(tmp_path: Path) -> None:
    path = tmp_path / "gate.json"
    path.write_text(gate(0.3).to_json({"x": 1.0}), encoding="utf-8")

    loaded = load_gate(path)

    assert loaded.threshold == 0.3
    assert loaded.weights.tolist() == [4.0]
    assert json.loads(path.read_text())["metrics"] == {"x": 1.0}


def test_stack_skips_empty() -> None:
    merged = stack([np.ones((2, 3)), np.zeros(0), np.zeros((1, 3))])
    assert merged.shape == (3, 3)
