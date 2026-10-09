from pathlib import Path
from typing import Any, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    # Опечатка в имени параметра конфига — ошибка, а не молча проигнорированное поле
    model_config = ConfigDict(extra="forbid", frozen=True)


class SplitConfig(StrictModel):
    train: float = Field(gt=0, lt=1)
    val: float = Field(gt=0, lt=1)
    test: float = Field(gt=0, lt=1)

    @model_validator(mode="after")
    def check_sum(self) -> Self:
        if abs(self.train + self.val + self.test - 1) > 1e-9:
            raise ValueError("Доли train, val и test должны в сумме давать 1")
        return self


class PathsConfig(StrictModel):
    raw: Path
    processed: Path
    artifacts: Path


class ExperimentConfig(StrictModel):
    name: str
    seed: int = Field(ge=0)
    image_size: int = Field(gt=0)
    batch_size: int = Field(gt=0)
    epochs: int = Field(gt=0)
    learning_rate: float = Field(gt=0)
    classes: tuple[str, ...] = Field(min_length=2)
    split: SplitConfig
    paths: PathsConfig

    @model_validator(mode="after")
    def check_unique_classes(self) -> Self:
        if len(set(self.classes)) != len(self.classes):
            raise ValueError("Классы не должны повторяться")
        return self


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _read(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: ожидался словарь параметров")
    parent = data.pop("extends", None)
    if parent is None:
        return data
    return _merge(_read(path.parent / parent), data)


def load_config(path: str | Path) -> ExperimentConfig:
    """Читает конфиг эксперимента. Поле extends подключает родительский конфиг,
    вложенные разделы объединяются, а не заменяются целиком."""
    return ExperimentConfig.model_validate(_read(Path(path)))
