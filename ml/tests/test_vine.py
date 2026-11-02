import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from vitiguard_ml.data.vine import block_split, is_original, prepare_inesc, voc_to_yolo
from vitiguard_ml.robot.vine import accept, best_entry, class_scores

VOC = b"""<annotation><size><width>100</width><height>50</height></size>
<object><name>trunk</name><bndbox><xmin>10</xmin><ymin>5</ymin><xmax>30</xmax><ymax>45</ymax></bndbox></object>
<object><name>tiny_grape_bunch</name><bndbox><xmin>60</xmin><ymin>0</ymin><xmax>120</xmax><ymax>10</ymax></bndbox></object>
<object><name>truk</name><bndbox><xmin>1</xmin><ymin>1</ymin><xmax>1.5</xmax><ymax>9</ymax></bndbox></object>
<object><name>leaf</name><bndbox><xmin>1</xmin><ymin>1</ymin><xmax>9</xmax><ymax>9</ymax></bndbox></object>
</annotation>"""


def test_only_original_frames() -> None:
    assert is_original("aveleda0001")
    assert is_original("left0088")
    for copy in ("flip_aveleda0040", "angle-30_aveleda_1", "randomcombination3_left1"):
        assert not is_original(copy)
    assert not is_original("thermal_aveleda098")


def test_voc_boxes_become_yolo_and_clip_to_frame() -> None:
    lines = voc_to_yolo(VOC)

    assert lines == [
        "0 0.200000 0.500000 0.200000 0.800000",
        "1 0.800000 0.100000 0.400000 0.200000",
    ]


def test_split_by_blocks_of_neighbouring_frames() -> None:
    names = [f"aveleda{index:04d}" for index in range(120)] + ["left0001"]

    splits = block_split(names)

    assert {splits[f"aveleda{index:04d}"] for index in range(50)} == {"test"}
    assert {splits[f"aveleda{index:04d}"] for index in range(50, 100)} == {"val"}
    assert splits["aveleda0100"] == "train"
    assert splits["left0001"] == "test"


def test_prepare_inesc_skips_copies(tmp_path: Path) -> None:
    archive = tmp_path / "data.zip"
    with zipfile.ZipFile(archive, "w") as zip_file:
        for name in ("aveleda0000", "flip_aveleda0000"):
            zip_file.writestr(f"all/{name}.xml", VOC)
            zip_file.writestr(f"all/{name}.jpg", b"jpeg")

    counts = prepare_inesc(archive, tmp_path / "out")

    assert counts == {"train": 0, "val": 0, "test": 1}
    assert (tmp_path / "out/images/test/inesc_aveleda0000.jpg").read_bytes() == b"jpeg"
    label = (tmp_path / "out/labels/test/inesc_aveleda0000.txt").read_text()
    assert label.startswith("0 ")


def test_class_scores_from_ultralytics_metrics() -> None:
    box = SimpleNamespace(
        map=0.5, map50=0.8, ap_class_index=np.array([0, 1]), maps=np.array([0.6, 0.4])
    )

    assert class_scores(SimpleNamespace(box=box)) == {
        "map50_95": 0.5,
        "map50": 0.8,
        "trunk_map50_95": 0.6,
        "bunch_map50_95": 0.4,
    }


def test_new_version_accepted_only_if_better_without_class_drop() -> None:
    best = {"map50_95": 0.5, "trunk_map50_95": 0.6, "bunch_map50_95": 0.4}

    assert accept(best, None)
    assert not accept({**best, "map50_95": 0.5}, best)
    better = {"map50_95": 0.55, "trunk_map50_95": 0.7, "bunch_map50_95": 0.395}
    assert accept(better, best)
    worse_class = {"map50_95": 0.55, "trunk_map50_95": 0.75, "bunch_map50_95": 0.35}
    assert not accept(worse_class, best)


def test_best_entry_is_last_accepted() -> None:
    registry = [
        {"version": "v001", "accepted": True},
        {"version": "v002", "accepted": False},
    ]

    assert best_entry(registry) == registry[0]
    assert best_entry([]) is None
