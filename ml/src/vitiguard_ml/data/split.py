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
    rows: list[ManifestRow], ratios: SplitConfig, seed: int
) -> dict[str, str]:
    """Возвращает выборку для каждого пути.

    Стратификация по паре (класс, источник): в каждую выборку попадает та же
    доля каждого класса из каждого источника. Иначе, например, почти все
    здоровые листья из NGLD могли бы оказаться в обучении, а в тесте — только
    из PlantVillage, и оценка качества была бы искажена. Повторы уже удалены
    при объединении, поэтому копии одного снимка в разные выборки не попадут."""
    strata: dict[tuple[str, str], list[str]] = defaultdict(list)
    for row in sorted(rows, key=lambda r: r.path):
        strata[(row.label, row.source)].append(row.path)

    generator = random.Random(seed)
    assignment: dict[str, str] = {}
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    parser.add_argument("--processed", type=Path, default=Path("data/processed"))
    args = parser.parse_args()

    config = load_config(args.config)
    rows = read_manifest(args.processed / "manifest.csv")
    assignment = split_rows(rows, config.split, config.seed)
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
