from dataclasses import replace
from pathlib import Path

from vitiguard_ml.data.combine import DEFAULT_SOURCES, combine
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


def test_drop_conflicts_removes_both_copies(tmp_path: Path) -> None:
    write_manifest(
        [
            replace(row("a", "healthy", "s1", "d1"), source="first"),
            replace(row("b", "healthy", "s2", "d2"), source="first"),
        ],
        tmp_path / "first" / "manifest.csv",
    )
    write_manifest(
        [replace(row("a", "esca", "s1", "d1"), source="second")],
        tmp_path / "second" / "manifest.csv",
    )

    rows, report = combine(tmp_path, ["first", "second"], drop_conflicts=True)

    assert [r.path for r in rows] == ["first/healthy/b.jpg"]
    assert report.dropped_conflicts == 1
    assert report.kept[("healthy", "first")] == 1


def test_exclude_skips_class_of_source(tmp_path: Path) -> None:
    write_manifest(
        [
            replace(row("a", "healthy", "s1", "d1"), source="first"),
            replace(row("b", "esca", "s2", "d2"), source="first"),
        ],
        tmp_path / "first" / "manifest.csv",
    )

    rows, report = combine(
        tmp_path, ["first"], exclude=frozenset({("first", "healthy")})
    )

    assert [r.label for r in rows] == ["esca"]
    assert report.kept[("healthy", "first")] == 0


def test_default_sources_includes_gvlid() -> None:
    assert DEFAULT_SOURCES == ["plantvillage", "ngld", "gvlid"]


def test_combine_copies_source_vineyards_with_prefix(tmp_path: Path) -> None:
    write_manifest(
        [replace(row("leaf1", "black_rot", "sha1", "dh1"), source="gvlid")],
        tmp_path / "gvlid" / "manifest.csv",
    )
    (tmp_path / "gvlid" / "vineyards.csv").write_text(
        "path,vineyard\nblack_rot/leaf1.jpg,Vineyard_05\n", encoding="utf-8"
    )

    rows, _ = combine(tmp_path, ["gvlid"])

    assert len(rows) == 1
    assert rows[0].path == "gvlid/black_rot/leaf1.jpg"
    combined_vineyards = tmp_path / "vineyards.csv"
    assert combined_vineyards.exists()
    content = combined_vineyards.read_text(encoding="utf-8").splitlines()
    assert content[0] == "path,vineyard"
    assert content[1] == "gvlid/black_rot/leaf1.jpg,Vineyard_05"
