from pathlib import Path

import pytest
from PIL import Image

from vitiguard_ml.mask_review import main, pick, summarize


def manifest(root: Path) -> Path:
    rows = ["path,label,source"]
    for label in ("healthy", "downy_mildew"):
        for index in range(4):
            path = f"{label}/{index}.png"
            (root / label).mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (64, 64), (60, 140, 50)).save(root / path)
            rows.append(f"{path},{label},test")
    (root / "manifest.csv").write_text("\n".join(rows), encoding="utf-8")
    return root / "manifest.csv"


def test_pick_takes_classes_evenly(tmp_path: Path) -> None:
    rows = pick(manifest(tmp_path), count=4)

    assert sorted(row["label"] for row in rows) == [
        "downy_mildew",
        "downy_mildew",
        "healthy",
        "healthy",
    ]
    assert pick(manifest(tmp_path), count=4) == rows


def test_summary_by_class(tmp_path: Path) -> None:
    answers = tmp_path / "answers.csv"
    answers.write_text(
        "path,label,severity,verdict\n"
        "a,healthy,0,correct\n"
        "b,healthy,0,wrong\n"
        "c,downy_mildew,0.2,partly\n"
        "d,downy_mildew,0.1,\n",
        encoding="utf-8",
    )

    result = summarize(answers)

    assert result["all"]["checked"] == 3
    assert result["healthy"]["correct"] == 0.5
    assert result["downy_mildew"] == {
        "checked": 1.0,
        "correct": 0.0,
        "partly": 1.0,
        "wrong": 0.0,
    }


def test_page_has_every_picked_leaf(tmp_path: Path) -> None:
    pytest.importorskip("cv2")
    manifest(tmp_path / "data")

    main(["page", "--data", str(tmp_path / "data"), "--out", str(tmp_path / "out")])

    page = (tmp_path / "out" / "index.html").read_text(encoding="utf-8")
    assert page.count("data:image/jpeg;base64,") == 8
    assert "answers.csv" in page
