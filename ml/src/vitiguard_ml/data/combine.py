"""Объединение подготовленных наборов в один манифест с едиными классами.

Запуск из каталога ml:  uv run python -m vitiguard_ml.data.combine
"""

import argparse
from collections import Counter
from dataclasses import dataclass, field, replace
from pathlib import Path

from vitiguard_ml.data.manifest import ManifestRow, read_manifest, write_manifest

# Порядок важен: при совпадении снимков остаётся экземпляр из источника выше
DEFAULT_SOURCES = ["plantvillage", "ngld"]


@dataclass
class CombineReport:
    kept: Counter[tuple[str, str]] = field(default_factory=Counter)
    exact_duplicates: int = 0
    near_duplicates: int = 0
    conflicting_labels: list[tuple[str, str]] = field(default_factory=list)


def combine(
    processed_root: Path, sources: list[str]
) -> tuple[list[ManifestRow], CombineReport]:
    """Пути в общем манифесте — относительно processed_root (<источник>/<класс>/...).

    Повторы ищутся по SHA-256 (точная копия) и по dHash (тот же снимок, но
    пересжатый или уменьшенный). Если копии одного снимка размечены разными
    классами, это записывается в отчёт: такие снимки нужно проверить вручную."""
    report = CombineReport()
    by_sha: dict[str, ManifestRow] = {}
    by_dhash: dict[str, ManifestRow] = {}
    rows: list[ManifestRow] = []
    for source in sources:
        for row in read_manifest(processed_root / source / "manifest.csv"):
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
    return rows, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed", type=Path, default=Path("data/processed"))
    parser.add_argument("--sources", nargs="+", default=DEFAULT_SOURCES)
    args = parser.parse_args()

    rows, report = combine(args.processed, args.sources)
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


if __name__ == "__main__":
    main()
