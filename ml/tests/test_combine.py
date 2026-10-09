from dataclasses import replace
from pathlib import Path

from vitiguard_ml.data.combine import combine
from vitiguard_ml.data.manifest import ManifestRow, write_manifest


def row(name: str, label: str, sha: str, dhash: str) -> ManifestRow:
    return ManifestRow(f"{label}/{name}.jpg", label, "", sha, dhash, 256, 256)


def test_sources_are_merged_and_duplicates_dropped(tmp_path: Path) -> None:
    write_manifest(
        [
            replace(row("a", "healthy", "s1", "d1"), source="first"),
            replace(row("b", "black_rot", "s2", "d2"), source="first"),
        ],
        tmp_path / "first" / "manifest.csv",
    )
    write_manifest(
        [
            replace(row("a2", "healthy", "s1", "d1"), source="second"),  # точная копия
            replace(
                row("b2", "black_rot", "s9", "d2"), source="second"
            ),  # пересжатая копия
            replace(row("c", "downy_mildew", "s3", "d3"), source="second"),
        ],
        tmp_path / "second" / "manifest.csv",
    )

    rows, report = combine(tmp_path, ["first", "second"])

    assert sorted(r.path for r in rows) == [
        "first/black_rot/b.jpg",
        "first/healthy/a.jpg",
        "second/downy_mildew/c.jpg",
    ]
    assert report.exact_duplicates == 1
    assert report.near_duplicates == 1
    assert report.conflicting_labels == []
    assert report.kept[("downy_mildew", "second")] == 1


def test_copies_with_different_labels_are_reported(tmp_path: Path) -> None:
    write_manifest(
        [replace(row("a", "healthy", "s1", "d1"), source="first")],
        tmp_path / "first" / "manifest.csv",
    )
    write_manifest(
        [replace(row("a", "esca", "s1", "d1"), source="second")],
        tmp_path / "second" / "manifest.csv",
    )

    rows, report = combine(tmp_path, ["first", "second"])

    assert len(rows) == 1
    assert report.conflicting_labels == [("first/healthy/a.jpg", "second/esca/a.jpg")]
