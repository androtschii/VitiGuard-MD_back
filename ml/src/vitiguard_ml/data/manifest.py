import csv
import hashlib
from dataclasses import asdict, dataclass, fields
from pathlib import Path


@dataclass(frozen=True)
class ManifestRow:
    """Одно изображение набора данных. Путь — относительно корня набора."""

    path: str
    label: str
    source: str
    sha256: str
    width: int
    height: int


COLUMNS = [field.name for field in fields(ManifestRow)]


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_manifest(rows: list[ManifestRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(asdict(row) for row in sorted(rows, key=lambda r: r.path))


def read_manifest(path: Path) -> list[ManifestRow]:
    with path.open(newline="", encoding="utf-8") as file:
        return [
            ManifestRow(
                path=row["path"],
                label=row["label"],
                source=row["source"],
                sha256=row["sha256"],
                width=int(row["width"]),
                height=int(row["height"]),
            )
            for row in csv.DictReader(file)
        ]
