import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from app.ml.classifier import IMAGENET_MEAN, IMAGENET_STD


@dataclass(frozen=True)
class ModelFile:
    """Файл классификатора для загрузки в браузер (офлайн-диагностика)."""

    name: str
    classes: list[str]
    image_size: int
    mean: list[float]
    std: list[float]
    # Начало SHA-256 файла: клиент сравнивает его с сохранённой копией
    version: str
    size_bytes: int


@lru_cache(maxsize=4)
def _describe(path: Path, mtime_ns: int, size: int) -> ModelFile:
    meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return ModelFile(
        name=meta["name"],
        classes=meta["classes"],
        image_size=meta["image_size"],
        mean=[round(float(v), 3) for v in IMAGENET_MEAN],
        std=[round(float(v), 3) for v in IMAGENET_STD],
        version=digest.hexdigest()[:16],
        size_bytes=size,
    )


def describe_model(path: Path) -> ModelFile:
    """Описание модели; хэш 16 МБ считается один раз, пока файл не изменится."""
    stat = path.stat()
    return _describe(path, stat.st_mtime_ns, stat.st_size)
