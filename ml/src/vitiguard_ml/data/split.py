"""Разбиение общего набора на train/val/test и статистика классов.

Запуск из каталога ml:  uv run python -m vitiguard_ml.data.split
"""

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

from vitiguard_ml.config import SplitConfig, load_config
from vitiguard_ml.data.manifest import COLUMNS, ManifestRow, read_manifest

SPLITS = ("train", "val", "test")


def split_rows(
    rows: list[ManifestRow],
    ratios: SplitConfig,
    seed: int,
    groups: dict[str, str] | None = None,
) -> dict[str, str]:
    """Возвращает выборку для каждого пути.

    Если задан словарь groups, все снимки с одинаковой группой попадают в одну
    выборку (детерминированное распределение групп по seed в долях ratios).
    Остальные строки распределяются со стратификацией по паре (класс, источник)."""
    grouped_paths: dict[str, list[str]] = defaultdict(list)
    ungrouped_rows: list[ManifestRow] = []

    for row in sorted(rows, key=lambda r: r.path):
        group = groups.get(row.path) if groups else None
        if group:
            grouped_paths[group].append(row.path)
        else:
            ungrouped_rows.append(row)

    assignment: dict[str, str] = {}

    if grouped_paths:
        group_generator = random.Random(seed)
        group_names = sorted(grouped_paths.keys())
        group_generator.shuffle(group_names)

        total_grouped = sum(len(paths) for paths in grouped_paths.values())
        targets = {
            "train": total_grouped * ratios.train,
            "val": total_grouped * ratios.val,
            "test": total_grouped * ratios.test,
        }
        assigned_counts: dict[str, int] = {"train": 0, "val": 0, "test": 0}

        for group in group_names:
            paths = grouped_paths[group]
            size = len(paths)
            chosen_split = max(
                ["train", "val", "test"],
                key=lambda s: targets[s] - assigned_counts[s],
            )
            assigned_counts[chosen_split] += size
            for path in paths:
                assignment[path] = chosen_split

    if ungrouped_rows:
        strata: dict[tuple[str, str], list[str]] = defaultdict(list)
        for row in sorted(ungrouped_rows, key=lambda r: r.path):
            strata[(row.label, row.source)].append(row.path)

        generator = random.Random(seed)
        for key in sorted(strata):
            paths = strata[key]
            generator.shuffle(paths)
            n_train = round(len(paths) * ratios.train)
            n_val = round(len(paths) * ratios.val)
            for index, path in enumerate(paths):
                if index < n_train:
                    assignment[path] = "train"
                elif index < n_train + n_val:
                    assignment[path] = "val"
                else:
                    assignment[path] = "test"

    return assignment


def class_weights(counts: dict[str, int], classes: tuple[str, ...]) -> dict[str, float]:
    """Веса классов для функции потерь, схема balanced из scikit-learn: вес
    обратно пропорционален числу примеров класса, а средний вес по всем примерам
    равен 1. Редкий класс (оидиум) весит больше, и модель не может выигрывать в
    точности, просто игнорируя его."""
    present = {label: counts[label] for label in classes if counts.get(label)}
    total = sum(present.values())
    return {
        label: round(total / (len(present) * count), 4)
        for label, count in present.items()
    }


def write_splits(
    rows: list[ManifestRow], assignment: dict[str, str], path: Path
) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=[*COLUMNS, "split"])
        writer.writeheader()
        for row in sorted(rows, key=lambda r: r.path):
            writer.writerow({**asdict(row), "split": assignment[row.path]})


@dataclass(frozen=True)
class SplitStats:
    counts: dict[str, dict[str, int]]
    imbalance_ratio: float
    class_weights: dict[str, float]


def build_stats(
    rows: list[ManifestRow], assignment: dict[str, str], classes: tuple[str, ...]
) -> SplitStats:
    per_split: dict[str, Counter[str]] = {name: Counter() for name in SPLITS}
    for row in rows:
        per_split[assignment[row.path]][row.label] += 1
    train = dict(per_split["train"])
    return SplitStats(
        counts={
            name: dict(sorted(counter.items())) for name, counter in per_split.items()
        },
        imbalance_ratio=round(max(train.values()) / min(train.values()), 2),
        class_weights=class_weights(train, classes),
    )


def read_groups(processed_root: Path, rows: list[ManifestRow]) -> dict[str, str]:
    """Читает vineyards.csv источников и возвращает группы вида <source>:<vineyard>."""
    groups: dict[str, str] = {}
    row_sources = {row.path: row.source for row in rows}

    combined_vineyards = processed_root / "vineyards.csv"
    if combined_vineyards.exists():
        with combined_vineyards.open(newline="", encoding="utf-8") as file:
            for entry in csv.DictReader(file):
                path = entry.get("path")
                vineyard = entry.get("vineyard")
                if path and vineyard:
                    source = row_sources.get(path, path.split("/")[0])
                    groups[path] = f"{source}:{vineyard}"

    sources = {row.source for row in rows}
    for source in sources:
        source_vineyards = processed_root / source / "vineyards.csv"
        if source_vineyards.exists():
            with source_vineyards.open(newline="", encoding="utf-8") as file:
                for entry in csv.DictReader(file):
                    path = entry.get("path")
                    vineyard = entry.get("vineyard")
                    if path and vineyard:
                        full_path = path if path in row_sources else f"{source}/{path}"
                        if full_path in row_sources and full_path not in groups:
                            groups[full_path] = f"{source}:{vineyard}"

    return groups


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    parser.add_argument("--processed", type=Path, default=Path("data/processed"))
    args = parser.parse_args()

    config = load_config(args.config)
    rows = read_manifest(args.processed / "manifest.csv")
    groups = read_groups(args.processed, rows)
    assignment = split_rows(rows, config.split, config.seed, groups=groups or None)
    write_splits(rows, assignment, args.processed / "splits.csv")
    stats = build_stats(rows, assignment, config.classes)
    (args.processed / "class_stats.json").write_text(
        json.dumps(
            {"seed": config.seed, **asdict(stats)}, ensure_ascii=False, indent=2
        ),
        encoding="utf-8",
    )

    print(f"{'класс':16}{'train':>8}{'val':>8}{'test':>8}{'вес':>8}")
    for label in config.classes:
        values = [stats.counts[name].get(label, 0) for name in SPLITS]
        weight = stats.class_weights.get(label, 0)
        print(f"{label:16}" + "".join(f"{v:>8}" for v in values) + f"{weight:>8}")
    print(f"всего: {len(rows)}, дисбаланс (макс/мин в train): {stats.imbalance_ratio}")


if __name__ == "__main__":
    main()
