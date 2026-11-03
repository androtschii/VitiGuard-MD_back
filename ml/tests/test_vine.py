import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from conftest import make_image
from PIL import Image

from vitiguard_ml.data.images import difference_hash
from vitiguard_ml.data.vine import (
    SOURCES,
    _hamming,
    _humain_labelimg_to_yolo,
    _humain_original_images,
    block_split,
    check_near_duplicates,
    collect_test_hashes,
    is_original,
    missing_sources,
    prepare_humain,
    prepare_inesc,
    prepare_inesc_second_camera,
    voc_to_yolo,
    write_config,
)
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
    assert is_original("rotated_left0151")


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


def test_second_camera_is_a_separate_source_without_test(tmp_path: Path) -> None:
    archive = tmp_path / "data.zip"
    with zipfile.ZipFile(archive, "w") as zip_file:
        for name in ["aveleda0000"] + [f"rotated_left{i:04d}" for i in range(60)]:
            zip_file.writestr(f"all/{name}.xml", VOC)
            zip_file.writestr(f"all/{name}.jpg", b"jpeg")

    first = prepare_inesc(archive, tmp_path / "out")
    second = prepare_inesc_second_camera(archive, tmp_path / "out")

    assert first == {"train": 0, "val": 0, "test": 1}
    assert second == {"train": 50, "val": 10, "test": 0}
    assert (tmp_path / "out/images/val/inesc_rotated_left0050.jpg").exists()


def test_only_missing_sources_are_built(tmp_path: Path) -> None:
    assert missing_sources(tmp_path) == list(SOURCES)

    write_config(
        tmp_path,
        {
            "inesc": {"test": 1},
            "wgisd": {"test": 1},
            "inesc_second_camera": {"val": 1},
            "humain": {"train": 1},
        },
    )

    assert missing_sources(tmp_path) == ["mendeley"]


def test_hamming_distance_calculation() -> None:
    assert _hamming("0000", "0000") == 0
    assert _hamming("0001", "0000") == 1
    assert _hamming("ffff", "0000") == 16


def test_near_duplicates_detection_by_dhash(tmp_path: Path) -> None:
    img_orig = tmp_path / "orig.jpg"
    make_image(img_orig, seed=42)
    img_copy = tmp_path / "copy.jpg"
    with Image.open(img_orig) as im:
        im.resize((128, 96)).save(img_copy, quality=60)
    img_diff = tmp_path / "diff.jpg"
    make_image(img_diff, seed=99)

    h_orig = difference_hash(Image.open(img_orig).convert("RGB"))
    dups = check_near_duplicates([img_copy, img_diff], {h_orig})

    assert img_copy in dups
    assert img_diff not in dups


def test_humain_labels_to_yolo_trunk(tmp_path: Path) -> None:
    label_file = tmp_path / "frame.txt"
    label_file.write_text("1 0.45 0.55 0.20 0.80\n2 0.10 0.20 0.15 0.30\n")

    lines = _humain_labelimg_to_yolo(label_file)

    assert lines == [
        "0 0.45 0.55 0.20 0.80",
        "0 0.10 0.20 0.15 0.30",
    ]


def test_humain_filters_augmented_frames(tmp_path: Path) -> None:
    root = tmp_path / "humain"
    img_dir = root / "Training_Set" / "images"
    gt_dir = root / "Training_Set" / "ground_truth"
    img_dir.mkdir(parents=True)
    gt_dir.mkdir(parents=True)

    make_image(img_dir / "trunk_001.jpg", seed=1)
    (gt_dir / "trunk_001.txt").write_text("0 0.5 0.5 0.2 0.8")

    make_image(img_dir / "rotate_trunk_001.jpg", seed=1)
    (gt_dir / "rotate_trunk_001.txt").write_text("0 0.5 0.5 0.2 0.8")

    make_image(img_dir / "blur_trunk_001.jpg", seed=1)
    (gt_dir / "blur_trunk_001.txt").write_text("0 0.5 0.5 0.2 0.8")

    pairs = _humain_original_images(root)

    assert len(pairs) == 1
    assert pairs[0][0].name == "trunk_001.jpg"


def test_collect_test_hashes_reads_images(tmp_path: Path) -> None:
    assert collect_test_hashes(tmp_path) == set()

    test_dir = tmp_path / "images" / "test"
    test_dir.mkdir(parents=True)
    img = test_dir / "sample.jpg"
    make_image(img, seed=7)

    hashes = collect_test_hashes(tmp_path)
    assert len(hashes) == 1


def test_prepare_humain_splits_into_train_and_val_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = tmp_path / "raw"
    root = raw / "humain" / "extracted" / "vine-trunk-main"
    img_dir = root / "Training_Set" / "images"
    gt_dir = root / "Training_Set" / "ground_truth"
    img_dir.mkdir(parents=True)
    gt_dir.mkdir(parents=True)

    for i in range(10):
        make_image(img_dir / f"frame_{i:03d}.jpg", seed=i)
        (gt_dir / f"frame_{i:03d}.txt").write_text("0 0.5 0.5 0.2 0.8")

    monkeypatch.setattr("vitiguard_ml.data.vine.download_humain", lambda _: root)

    out = tmp_path / "out"
    counts = prepare_humain(raw, out)

    assert counts["test"] == 0
    assert counts["train"] + counts["val"] == 10
    assert counts["val"] == 2
    assert not (out / "images" / "test").exists()


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
