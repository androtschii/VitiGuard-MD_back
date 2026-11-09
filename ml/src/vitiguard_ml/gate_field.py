"""Фильтр «лист / не лист» видит и снимки куста (ml/fine-01).

Прежний фильтр (ml/pr-021) учился на листьях по одному: лабораторных
(PlantVillage, NGLD) и вырезках HERMOS вокруг одного пятна. Снимок куста с
несколькими листьями, особенно маленький из мессенджера, он отсеивал как «не
виноград». Здесь в «виноград» добавляются целые полевые кадры: снимки HERMOS и
Tardif (Flavescence dorée, эска) и ряды INESC. Кадры уменьшаются до размеров
фото с телефона и из мессенджера, часть их откладывается для проверки. Модель
болезней не переобучается: меняется только фильтр поверх её признаков.

    python -m vitiguard_ml.gate_field --model diagnosis.onnx
        --lab <листья по одному> --crops <вырезки HERMOS>
        --canopy hermos=<кадры> tardif=<кадры> inesc=<кадры>
        --not-grape imagenette=<папка> crops=<листья других культур>
        --out <папка результата>"""

import argparse
import json
import random
import shutil
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from vitiguard_ml.leaf_gate import (
    Gate,
    add_features_output,
    extract,
    fit_gate,
    images,
    with_threshold,
)

# Длинная сторона: фото с телефона после сжатия и фото из мессенджера
SIDES = (1024, 320)
CHECK_SHARE = 0.3


def shrink_frames(
    sources: Sequence[Path], out: Path, limit: int, rng: random.Random
) -> list[Path]:
    """Кадры куста в двух размерах: большой снимок и маленький, как после
    пересылки. Нечитаемые файлы пропускаются."""
    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for index, path in enumerate(images(sources, limit, rng)):
        try:
            with Image.open(path) as image:
                frame = image.convert("RGB")
        except OSError:
            continue
        for side in SIDES:
            copy = frame.copy()
            copy.thumbnail((side, side), Image.Resampling.LANCZOS)
            target = out / f"{index:05d}_{side}.jpg"
            copy.save(target, quality=80)
            written.append(target)
    return written


def frame_key(path: Path) -> str:
    """Оба размера одного кадра — один ключ: при разбиении они не расходятся
    в обучение и проверку."""
    return path.stem.rsplit("_", 1)[0]


def split_by_frame(
    paths: Sequence[Path], share: float, rng: random.Random
) -> tuple[list[Path], list[Path]]:
    keys = sorted({frame_key(path) for path in paths})
    rng.shuffle(keys)
    check_keys = set(keys[: round(len(keys) * share)])
    train = [path for path in paths if frame_key(path) not in check_keys]
    check = [path for path in paths if frame_key(path) in check_keys]
    return train, check


def paper_images(out: Path, count: int, rng: random.Random) -> list[Path]:
    """Синтетическая бумага: светлый фон, строки, тень, размытие. Белая
    поверхность похожа на налёт оидиума — фильтр должен её отсеивать."""
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for index in range(count):
        tone = rng.randint(200, 250)
        tint = tuple(min(255, tone + rng.randint(-12, 6)) for _ in range(3))
        image = Image.new("RGB", (320, 320), tint)
        draw = ImageDraw.Draw(image)
        for _ in range(rng.randint(0, 12)):
            y = rng.randint(10, 310)
            shade = tone - rng.randint(60, 160)
            draw.line(
                (rng.randint(0, 60), y, rng.randint(200, 320), y),
                fill=(shade, shade, shade),
                width=rng.randint(1, 3),
            )
        shadow = rng.randint(0, 80)
        draw.rectangle((0, 320 - shadow, 320, 320), fill=(tone - 40,) * 3)
        image = image.filter(ImageFilter.GaussianBlur(rng.uniform(0, 1.5)))
        target = out / f"paper_{index:04d}.jpg"
        image.save(target, quality=85)
        written.append(target)
    return written


def named_folders(values: Sequence[str]) -> dict[str, list[Path]]:
    """«имя=папка» → папки по имени; одно имя можно повторить."""
    named: dict[str, list[Path]] = {}
    for value in values:
        name, _, folder = value.partition("=")
        if not name or not folder:
            raise ValueError(f"нужно имя=папка: {value}")
        named.setdefault(name, []).append(Path(folder))
    return named


def pass_rate(gate: Gate, features: np.ndarray) -> float:
    if len(features) == 0:
        return float("nan")
    return round(float(np.mean(gate.scores(features) >= gate.threshold)), 4)


def report(
    gates: dict[str, Gate],
    grape_checks: dict[str, np.ndarray],
    other_checks: dict[str, np.ndarray],
) -> dict[str, dict[str, float]]:
    """Для каждого фильтра: доля пропущенного винограда по источникам и доля
    отсеянного «не винограда» по видам."""
    result: dict[str, dict[str, float]] = {}
    for name, gate in gates.items():
        row = {
            f"pass_{key}": pass_rate(gate, value) for key, value in grape_checks.items()
        }
        for key, value in other_checks.items():
            row[f"reject_{key}"] = round(1 - pass_rate(gate, value), 4)
        row["threshold"] = round(gate.threshold, 4)
        result[name] = row
    return result


def stack(arrays: Sequence[np.ndarray]) -> np.ndarray:
    """Наборы признаков в одну матрицу; пустые наборы пропускаются."""
    return np.concatenate([array for array in arrays if len(array)])


def load_gate(path: Path) -> Gate:
    data = json.loads(path.read_text(encoding="utf-8"))
    return Gate(
        np.asarray(data["mean"]),
        np.asarray(data["std"]),
        np.asarray(data["weights"]),
        float(data["bias"]),
        float(data["threshold"]),
    )


def main() -> None:  # pragma: no cover — работа с настоящей моделью и данными
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--lab", type=Path, nargs="+", required=True)
    parser.add_argument("--crops", type=Path, nargs="+", required=True)
    parser.add_argument("--canopy", nargs="+", required=True)
    parser.add_argument("--not-grape", nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--per-set", type=int, default=1200)
    parser.add_argument("--keep", type=float, default=0.97)
    args = parser.parse_args()

    rng = random.Random(17)
    work = args.out / "work"
    shutil.rmtree(work, ignore_errors=True)
    old_path = args.model.with_suffix(".gate.json")
    old_gate = load_gate(old_path) if old_path.exists() else None
    add_features_output(args.model, args.model)

    def features(paths: Sequence[Path]) -> np.ndarray:
        print(f"признаки: {len(paths)} снимков", flush=True)
        return extract(args.model, paths)

    lab = images(args.lab, args.per_set * 2, rng)
    crops = images(args.crops, args.per_set * 2, rng)
    grape_train = [features(lab[: args.per_set]), features(crops[: args.per_set])]
    grape_checks = {
        "lab": features(lab[args.per_set :]),
        "crops": features(crops[args.per_set :]),
    }
    for name, folders in named_folders(args.canopy).items():
        frames = shrink_frames(folders, work / name, args.per_set // 2, rng)
        train, check = split_by_frame(frames, CHECK_SHARE, rng)
        grape_train.append(features(train))
        grape_checks[f"canopy_{name}"] = features(check)

    not_named = named_folders(args.not_grape)
    paper_images(work / "paper", 600, rng)
    not_named["paper"] = [work / "paper"]
    other_train, other_checks = [], {}
    for name, folders in not_named.items():
        found = images(folders, args.per_set * 2, rng)
        split = len(found) * 3 // 4
        other_train.append(features(found[:split]))
        other_checks[name] = features(found[split:])

    field_check = stack([value for key, value in grape_checks.items() if key != "lab"])
    gate = with_threshold(
        fit_gate(stack(grape_train), stack(other_train)),
        field_check,
        args.keep,
    )
    gates = {"old": old_gate, "new": gate} if old_gate else {"new": gate}
    metrics = report(gates, grape_checks, other_checks)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "diagnosis.gate.json").write_text(
        gate.to_json(metrics["new"]), encoding="utf-8"
    )
    (args.out / "gate_report.json").write_text(
        json.dumps(metrics, indent=1, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=1, ensure_ascii=False))


if __name__ == "__main__":  # pragma: no cover
    main()
