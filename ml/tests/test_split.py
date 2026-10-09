from collections import Counter
from pathlib import Path

import pytest

from vitiguard_ml.config import SplitConfig
from vitiguard_ml.data.manifest import ManifestRow
from vitiguard_ml.data.split import build_stats, class_weights, split_rows, write_splits

RATIOS = SplitConfig(train=0.7, val=0.15, test=0.15)


def rows(label: str, source: str, count: int) -> list[ManifestRow]:
    return [
        ManifestRow(
            f"{source}/{label}/{i:04}.jpg", label, source, f"{label}{i}", "0", 1, 1
        )
        for i in range(count)
    ]


@pytest.fixture
def dataset() -> list[ManifestRow]:
    return [
        *rows("healthy", "plantvillage", 400),
        *rows("healthy", "ngld", 600),
        *rows("downy_mildew", "ngld", 1000),
        *rows("powdery_mildew", "ngld", 400),
    ]


def test_every_image_gets_exactly_one_split(dataset: list[ManifestRow]) -> None:
    assignment = split_rows(dataset, RATIOS, seed=42)

    assert set(assignment) == {row.path for row in dataset}
    assert set(assignment.values()) == {"train", "val", "test"}


@pytest.mark.parametrize(
    ("label", "source"),
    [("healthy", "plantvillage"), ("healthy", "ngld"), ("powdery_mildew", "ngld")],
)
def test_each_class_and_source_is_split_in_the_same_proportion(
    dataset: list[ManifestRow], label: str, source: str
) -> None:
    assignment = split_rows(dataset, RATIOS, seed=42)
    stratum = [r.path for r in dataset if r.label == label and r.source == source]

    counts = Counter(assignment[path] for path in stratum)

    assert counts["train"] / len(stratum) == pytest.approx(0.7, abs=0.01)
    assert counts["val"] / len(stratum) == pytest.approx(0.15, abs=0.01)


def test_split_is_reproducible_and_depends_on_seed(dataset: list[ManifestRow]) -> None:
    first = split_rows(dataset, RATIOS, seed=42)

    assert split_rows(list(reversed(dataset)), RATIOS, seed=42) == first
    assert split_rows(dataset, RATIOS, seed=7) != first


def test_rare_class_gets_larger_weight() -> None:
    weights = class_weights(
        {"healthy": 700, "downy_mildew": 700, "powdery_mildew": 280},
        ("healthy", "downy_mildew", "powdery_mildew", "esca"),
    )

    assert weights["powdery_mildew"] > weights["healthy"]
    assert weights["healthy"] == weights["downy_mildew"]
    # Средний вес по примерам — 1: суммарный вклад всех примеров в потери не меняется
    counts = {"healthy": 700, "downy_mildew": 700, "powdery_mildew": 280}
    weighted = sum(counts[label] * weights[label] for label in counts)
    assert weighted / sum(counts.values()) == pytest.approx(1)
    # Класса нет в обучающей выборке — веса нет, а не деление на ноль
    assert "esca" not in weights


def test_stats_count_imbalance_in_train(dataset: list[ManifestRow]) -> None:
    assignment = split_rows(dataset, RATIOS, seed=42)

    stats = build_stats(
        dataset, assignment, ("healthy", "downy_mildew", "powdery_mildew")
    )

    assert stats.counts["train"] == {
        "downy_mildew": 700,
        "healthy": 700,
        "powdery_mildew": 280,
    }
    assert stats.imbalance_ratio == 2.5


def test_splits_file_has_manifest_columns_and_split(
    dataset: list[ManifestRow], tmp_path: Path
) -> None:
    assignment = split_rows(dataset, RATIOS, seed=42)
    target = tmp_path / "splits.csv"

    write_splits(dataset, assignment, target)

    header = target.read_text(encoding="utf-8").splitlines()[0]
    assert header == "path,label,source,sha256,dhash,width,height,split"
