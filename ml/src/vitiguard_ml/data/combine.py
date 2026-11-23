"""Объединение подготовленных наборов в один манифест с едиными классами.

Запуск из каталога ml:  uv run python -m vitiguard_ml.data.combine
"""

import argparse
import csv
from collections import Counter
from dataclasses import dataclass, field, replace
from pathlib import Path

from vitiguard_ml.data.manifest import ManifestRow, read_manifest, write_manifest

# Порядок важен: при совпадении снимков остаётся экземпляр из источника выше
DEFAULT_SOURCES = ["plantvillage", "ngld", "gvlid"]


@dataclass
class CombineReport:
    kept: Counter[tuple[str, str]] = field(default_factory=Counter)
    exact_duplicates: int = 0
    near_duplicates: int = 0
    conflicting_labels: list[tuple[str, str]] = field(default_factory=list)
    dropped_conflicts: int = 0


def combine(
    processed_root: Path,
    sources: list[str],
    drop_conflicts: bool = False,
    exclude: frozenset[tuple[str, str]] = frozenset(),
) -> tuple[list[ManifestRow], CombineReport]:
    """Пути в общем манифесте — относительно processed_root (<источник>/<класс>/...).

    Повторы ищутся по SHA-256 (точная копия) и по dHash (тот же снимок, но
    пересжатый или уменьшенный). Если копии одного снимка размечены разными
    классами, это записывается в отчёт: такие снимки нужно проверить вручную.
    С drop_conflicts=True из набора убираются обе копии такого снимка: какая
    метка верна, неизвестно, а учить модель на противоречии нельзя.
    exclude — пары (источник, класс), которые в набор не берутся."""
    report = CombineReport()
    by_sha: dict[str, ManifestRow] = {}
    by_dhash: dict[str, ManifestRow] = {}
    rows: list[ManifestRow] = []
    for source in sources:
        for row in read_manifest(processed_root / source / "manifest.csv"):
            if (source, row.label) in exclude:
                continue
            row = replace(row, path=f"{source}/{row.path}")
            if (first := by_sha.get(row.sha256)) is not None:
                report.exact_duplicates += 1
            elif (first := by_dhash.get(row.dhash)) is not None:
                report.near_duplicates += 1
            else:
                by_sha[row.sha256] = row
                by_dhash[row.dhash] = row
                rows.append(row)
                report.kept[(row.label, row.source)] += 1
                continue
            if first.label != row.label:
                report.conflicting_labels.append((first.path, row.path))

    if drop_conflicts and report.conflicting_labels:
        ambiguous = {first for first, _ in report.conflicting_labels}
        for row in rows:
            if row.path in ambiguous:
                report.kept[(row.label, row.source)] -= 1
        rows = [row for row in rows if row.path not in ambiguous]
        report.dropped_conflicts = len(ambiguous)

    kept_paths = {row.path for row in rows}
    vineyard_rows: list[dict[str, str]] = []
    for source in sources:
        source_vineyards = processed_root / source / "vineyards.csv"
        if source_vineyards.exists():
            with source_vineyards.open(newline="", encoding="utf-8") as file:
                for entry in csv.DictReader(file):
                    path = entry.get("path")
                    vineyard = entry.get("vineyard")
                    if path and vineyard:
                        prefixed = f"{source}/{path}"
                        if prefixed in kept_paths:
                            vineyard_rows.append(
                                {"path": prefixed, "vineyard": vineyard}
                            )

    out_vineyards = processed_root / "vineyards.csv"
    if vineyard_rows:
        with out_vineyards.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=["path", "vineyard"])
            writer.writeheader()
            writer.writerows(sorted(vineyard_rows, key=lambda r: r["path"]))
    elif out_vineyards.exists():
        out_vineyards.unlink()

    return rows, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed", type=Path, default=Path("data/processed"))
    parser.add_argument("--sources", nargs="+", default=DEFAULT_SOURCES)
    parser.add_argument(
        "--drop-conflicts",
        action="store_true",
        help="убрать снимки, одна и та же копия которых размечена разными классами",
    )
    parser.add_argument(
        "--exclude",
        nargs="+",
        default=[],
        metavar="ИСТОЧНИК:КЛАСС",
        help="не брать класс источника, например gvlid:healthy",
    )
    args = parser.parse_args()

    exclude = frozenset(
        (source, label)
        for source, label in (item.split(":", 1) for item in args.exclude)
    )
    rows, report = combine(args.processed, args.sources, args.drop_conflicts, exclude)
    write_manifest(rows, args.processed / "manifest.csv")

    labels = sorted({label for label, _ in report.kept})
    print(
        f"{'класс':16}" + "".join(f"{s:>14}" for s in args.sources) + f"{'всего':>10}"
    )
    for label in labels:
        counts = [report.kept[(label, source)] for source in args.sources]
        print(
            f"{label:16}" + "".join(f"{c:>14}" for c in counts) + f"{sum(counts):>10}"
        )
    print(f"итого: {len(rows)}")
    print(
        f"точных повторов: {report.exact_duplicates}, похожих (dHash): {report.near_duplicates}"
    )
    print(f"повторов с разными классами: {len(report.conflicting_labels)}")
    if args.drop_conflicts:
        print(f"убрано спорных снимков: {report.dropped_conflicts}")


if __name__ == "__main__":
    main()
