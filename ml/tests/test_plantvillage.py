from pathlib import Path

import pytest
from PIL import Image

from vitiguard_ml.data.manifest import read_manifest, write_manifest
from vitiguard_ml.data.plantvillage import CLASS_MAP, prepare


def make_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 6), color).save(path)


@pytest.fixture
def raw(tmp_path: Path) -> Path:
    root = tmp_path / "raw"
    for index, folder in enumerate(CLASS_MAP):
        make_image(root / folder / "a.JPG", (index * 40, 10, 10))
        make_image(root / folder / "b.jpg", (index * 40, 20, 20))
    return root


def test_images_are_copied_under_project_class_codes(raw: Path, tmp_path: Path) -> None:
    rows, skipped = prepare(raw, tmp_path / "out")

    assert sorted({row.label for row in rows}) == [
        "black_rot",
        "esca",
        "healthy",
        "leaf_blight",
    ]
    assert len(rows) == 8
    assert not skipped
    for row in rows:
        assert (tmp_path / "out" / row.path).is_file()
        assert row.path.startswith(f"{row.label}/")
        assert row.path.endswith(".jpg")
        assert (row.width, row.height) == (8, 6)
        assert row.source == "plantvillage"


def test_duplicates_corrupt_and_foreign_files_are_skipped(
    raw: Path, tmp_path: Path
) -> None:
    healthy = raw / "Grape___healthy"
    (healthy / "copy.jpg").write_bytes((healthy / "a.JPG").read_bytes())
    (healthy / "broken.jpg").write_bytes(b"not an image")
    (healthy / "notes.txt").write_text("readme", encoding="utf-8")

    rows, skipped = prepare(raw, tmp_path / "out")

    assert len(rows) == 8
    assert skipped == {"дубликат": 1, "повреждён": 1, "не изображение": 1}


def test_missing_class_folder_is_an_error(raw: Path, tmp_path: Path) -> None:
    for file in (raw / "Grape___Black_rot").iterdir():
        file.unlink()
    (raw / "Grape___Black_rot").rmdir()

    with pytest.raises(FileNotFoundError, match="Grape___Black_rot"):
        prepare(raw, tmp_path / "out")


def test_manifest_roundtrip(raw: Path, tmp_path: Path) -> None:
    rows, _ = prepare(raw, tmp_path / "out")
    manifest = tmp_path / "out" / "manifest.csv"

    write_manifest(rows, manifest)

    assert read_manifest(manifest) == sorted(rows, key=lambda row: row.path)
