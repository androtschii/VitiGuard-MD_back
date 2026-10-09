from pathlib import Path

import pytest
from conftest import make_image

from vitiguard_ml.data.folders import import_folders
from vitiguard_ml.data.manifest import read_manifest, write_manifest
from vitiguard_ml.data.ngld import CLASS_MAP as NGLD_CLASSES
from vitiguard_ml.data.plantvillage import CLASS_MAP as PLANTVILLAGE_CLASSES

CLASS_MAP = {"Sick": "black_rot", "Fine": "healthy"}


@pytest.fixture
def raw(tmp_path: Path) -> Path:
    root = tmp_path / "raw"
    make_image(root / "Sick" / "a.JPG", seed=1)
    make_image(root / "Sick" / "b.jpg", seed=2)
    make_image(root / "Fine" / "c.png", seed=3)
    make_image(root / "Ignored" / "d.jpg", seed=4)
    return root


def test_folders_become_project_classes(raw: Path, tmp_path: Path) -> None:
    rows, skipped = import_folders(raw, CLASS_MAP, "test", tmp_path / "out")

    assert sorted(row.label for row in rows) == ["black_rot", "black_rot", "healthy"]
    assert not skipped
    for row in rows:
        assert (tmp_path / "out" / row.path).is_file()
        assert row.path.startswith(f"{row.label}/")
        assert row.path.endswith((".jpg", ".png"))
        assert (row.width, row.height) == (64, 48)
        assert len(row.dhash) == 16
        assert row.source == "test"


def test_duplicates_corrupt_and_foreign_files_are_skipped(
    raw: Path, tmp_path: Path
) -> None:
    sick = raw / "Sick"
    (sick / "copy.jpg").write_bytes((sick / "b.jpg").read_bytes())
    (sick / "broken.jpg").write_bytes(b"not an image")
    (sick / "desktop.ini").write_text("[.ShellClassInfo]", encoding="utf-8")

    rows, skipped = import_folders(raw, CLASS_MAP, "test", tmp_path / "out")

    assert len(rows) == 3
    assert skipped == {"дубликат": 1, "повреждён": 1, "не изображение": 1}


def test_missing_class_folder_is_an_error(raw: Path, tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Missing"):
        import_folders(raw, {**CLASS_MAP, "Missing": "esca"}, "test", tmp_path / "out")


def test_manifest_roundtrip(raw: Path, tmp_path: Path) -> None:
    rows, _ = import_folders(raw, CLASS_MAP, "test", tmp_path / "out")
    manifest = tmp_path / "out" / "manifest.csv"

    write_manifest(rows, manifest)

    assert read_manifest(manifest) == sorted(rows, key=lambda row: row.path)


def test_source_class_maps_use_project_classes() -> None:
    project = {
        "healthy",
        "black_rot",
        "esca",
        "leaf_blight",
        "downy_mildew",
        "powdery_mildew",
    }

    assert set(PLANTVILLAGE_CLASSES.values()) <= project
    assert set(NGLD_CLASSES.values()) == {"healthy", "downy_mildew", "powdery_mildew"}
