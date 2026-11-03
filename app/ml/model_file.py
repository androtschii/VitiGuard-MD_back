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


@dataclass(frozen=True)
class RobotModelFile:
    """Модель лозы (YOLO ONNX) для бортового компьютера робота."""

    name: str
    classes: list[str]
    image_size: int
    version: str
    size_bytes: int
    format: str = "onnx"


@lru_cache(maxsize=4)
def _describe_robot(path: Path, mtime_ns: int, size: int) -> RobotModelFile:
    meta_path = path.with_suffix(".json")
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        name = meta.get("name", "vine_yolo")
        classes = meta.get("classes", ["trunk", "bunch"])
        image_size = meta.get("image_size", 640)
    else:
        name = "vine_yolo"
        classes = ["trunk", "bunch"]
        image_size = 640

    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return RobotModelFile(
        name=name,
        classes=classes,
        image_size=image_size,
        version=digest.hexdigest()[:16],
        size_bytes=size,
        format="onnx",
    )


def describe_robot_model(path: Path) -> RobotModelFile:
    """Описание модели лозы для робота; хэш кэшируется по размеру и mtime."""
    stat = path.stat()
    return _describe_robot(path, stat.st_mtime_ns, stat.st_size)
