import json
from pathlib import Path

import pytest
from PIL import Image

from vitiguard_ml.data.gvlid import CLASS_MAP, SOURCE, prepare


def make_test_image(
    path: Path,
    color: tuple[int, int, int] = (50, 100, 150),
    size: tuple[int, int] = (1080, 1080),
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", size, color)
    img.save(path, format="JPEG")


def setup_raw_gvlid(raw: Path) -> None:
    for folder in CLASS_MAP:
        (raw / folder).mkdir(parents=True, exist_ok=True)


def test_class_map_matches_project_classes() -> None:
    assert SOURCE == "gvlid"
    assert CLASS_MAP == {
        "Black rot": "black_rot",
        "esca": "esca",
        "healthy": "healthy",
        "leaf blight": "leaf_blight",
    }


def test_label_from_folder_overrides_metadata_discrepancy(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    setup_raw_gvlid(raw)
    make_test_image(raw / "leaf blight" / "img001.jpg")
    metadata = [
        {
            "image_filename": "img001.jpg",
            "Plant_Health_Status": "Healthy",
            "Disease_Type": "Healthy",
            "Location": {"Vineyard": "Vineyard_1"},
        }
    ]
    (raw / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    out = tmp_path / "out"
    rows, skipped = prepare(raw, out)

    assert len(rows) == 1
    assert rows[0].label == "leaf_blight"
    assert not skipped


def test_image_resized_to_256x256(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    setup_raw_gvlid(raw)
    make_test_image(raw / "Black rot" / "img_large.jpg", size=(1080, 720))
    (raw / "metadata.json").write_text("[]", encoding="utf-8")

    out = tmp_path / "out"
    rows, _ = prepare(raw, out)

    assert len(rows) == 1
    row = rows[0]
    assert row.width == 256
    assert row.height == 256
    saved_path = out / row.path
    assert saved_path.exists()
    with Image.open(saved_path) as saved_img:
        assert saved_img.size == (256, 256)
        assert saved_img.format == "JPEG"
    assert len(row.sha256) == 64
    assert len(row.dhash) == 16


def test_vineyards_csv_generation(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    setup_raw_gvlid(raw)
    make_test_image(raw / "healthy" / "h1.jpg", color=(50, 100, 150))
    make_test_image(raw / "esca" / "e1.jpg", color=(150, 100, 50))
    metadata = [
        {"image_filename": "h1.jpg", "Location": {"Vineyard": "Vineyard_10"}},
        {"image_filename": "e1.jpg", "Location": {"Vineyard": "Vineyard_20"}},
    ]
    (raw / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    out = tmp_path / "out"
    rows, _ = prepare(raw, out)

    vineyards_file = out / "vineyards.csv"
    assert vineyards_file.exists()
    lines = vineyards_file.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "path,vineyard"
    mapping = dict(line.split(",") for line in lines[1:])
    row_healthy = next(r for r in rows if r.label == "healthy")
    row_esca = next(r for r in rows if r.label == "esca")
    assert mapping[row_healthy.path] == "Vineyard_10"
    assert mapping[row_esca.path] == "Vineyard_20"


def test_unregistered_image_warns_and_marks_unknown(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    raw = tmp_path / "raw"
    setup_raw_gvlid(raw)
    make_test_image(raw / "healthy" / "h_unknown.jpg")
    (raw / "metadata.json").write_text("[]", encoding="utf-8")

    out = tmp_path / "out"
    rows, _ = prepare(raw, out)

    assert len(rows) == 1
    captured = capsys.readouterr().out
    assert "unknown" in captured
    vineyards_content = (out / "vineyards.csv").read_text(encoding="utf-8")
    assert "unknown" in vineyards_content
