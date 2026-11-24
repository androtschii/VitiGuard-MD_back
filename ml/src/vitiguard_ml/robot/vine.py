"""Модель лозы: стволы и грозди (ml/pr-023), ежедневное дообучение (ml/pr-032).

Каждый запуск дообучает модель от лучших принятых весов и сравнивает новую
версию с принятой на одной и той же тестовой выборке. Новая версия
принимается, только если средняя точность (mAP50-95) выросла, а ни один класс
не просел больше чем на 1 пункт. Все запуски записываются в реестр
(ml/pr-033): дата, данные, метрики, принята или нет.

Запуск из каталога ml:
  uv run --group train python -m vitiguard_ml.robot.vine --epochs 5
"""

import argparse
import importlib
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from vitiguard_ml.data.vine import CLASSES

START_MODEL = "yolo11n.pt"
# Класс не должен проседать больше чем на 1 пункт mAP50-95
MAX_CLASS_DROP = 0.01
# Порог уверенности на сайте (app/ml/vine_detector.py)
SITE_CONFIDENCE = 0.25


def class_scores(metrics: Any) -> dict[str, float]:
    """mAP50-95 по классам и в среднем из метрик Ultralytics."""
    per_class = {
        CLASSES[int(index)]: round(float(value), 4)
        for index, value in zip(
            metrics.box.ap_class_index,
            metrics.box.maps[metrics.box.ap_class_index],
            strict=True,
        )
    }
    return {
        "map50_95": round(float(metrics.box.map), 4),
        "map50": round(float(metrics.box.map50), 4),
        **{f"{name}_map50_95": score for name, score in per_class.items()},
    }


def accept(new: dict[str, float], best: dict[str, float] | None) -> bool:
    """Правило выкатки: лучше в среднем и без заметной просадки по классу."""
    if best is None:
        return True
    if new["map50_95"] <= best["map50_95"]:
        return False
    return all(
        new.get(f"{name}_map50_95", 0.0)
        >= best.get(f"{name}_map50_95", 0.0) - MAX_CLASS_DROP
        for name in CLASSES
    )


def accept_with_negatives(
    new: dict[str, float],
    best: dict[str, float],
    new_false_rate: float,
    best_false_rate: float,
) -> bool:
    """Запуск с пустыми снимками (ml/fine-07) должен убрать ложные рамки, а не
    поднять mAP: принимается, если доля пустых снимков с рамками упала, средний
    mAP50-95 и каждый класс просели не больше чем на 1 пункт."""
    if new_false_rate >= best_false_rate:
        return False
    if new["map50_95"] < best["map50_95"] - MAX_CLASS_DROP:
        return False
    return all(
        new.get(f"{name}_map50_95", 0.0)
        >= best.get(f"{name}_map50_95", 0.0) - MAX_CLASS_DROP
        for name in CLASSES
    )


def false_rate(model: Any, images: list[Path], image_size: int) -> float:
    """Доля снимков без стволов и гроздей, на которых модель нашла хоть что-то
    при пороге сайта (0,25, app/ml/vine_detector.py)."""
    if not images:
        return 0.0
    found = 0
    for path in images:
        result = model.predict(
            str(path), imgsz=image_size, conf=SITE_CONFIDENCE, verbose=False
        )[0]
        found += int(len(result.boxes) > 0)
    return round(found / len(images), 4)


def load_registry(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    entries: list[dict[str, Any]] = json.loads(path.read_text(encoding="utf-8"))
    return entries


def best_entry(registry: list[dict[str, Any]]) -> dict[str, Any] | None:
    accepted = [entry for entry in registry if entry["accepted"]]
    return accepted[-1] if accepted else None


def run(args: argparse.Namespace) -> dict[str, Any]:
    yolo = importlib.import_module("ultralytics").YOLO
    out: Path = args.out
    registry_path = out / "registry.json"
    registry = load_registry(registry_path)
    best = best_entry(registry)
    start = best["weights"] if best else args.model
    version = len(registry) + 1
    name = f"v{version:03d}"

    model = yolo(start)
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.image_size,
        batch=args.batch,
        seed=42,
        deterministic=True,
        project=str(out / "runs"),
        name=name,
        exist_ok=True,
        workers=args.workers,
        device="cpu" if args.cpu else None,
    )
    trained = out / "runs" / name / "weights" / "best.pt"
    scores = class_scores(
        yolo(str(trained)).val(
            data=str(args.data), split="test", imgsz=args.image_size, plots=False
        )
    )
    negatives: dict[str, Any] | None = None
    if args.negatives is not None:
        images = sorted(
            path
            for path in Path(args.negatives).iterdir()
            if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        negatives = {
            "images": len(images),
            "false_rate": false_rate(yolo(str(trained)), images, args.image_size),
        }
        if best:
            negatives["best_false_rate"] = false_rate(
                yolo(best["weights"]), images, args.image_size
            )
    if negatives is not None and best and "best_false_rate" in negatives:
        accepted = accept_with_negatives(
            scores,
            best["test"],
            negatives["false_rate"],
            negatives["best_false_rate"],
        )
    else:
        accepted = accept(scores, best["test"] if best else None)
    weights = out / "models" / f"{name}.pt"
    weights.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(trained, weights)
    entry: dict[str, Any] = {
        "version": name,
        "date": datetime.now(UTC).isoformat(timespec="seconds"),
        "started_from": str(start),
        "epochs": args.epochs,
        "image_size": args.image_size,
        "data": str(args.data),
        "test": scores,
        "accepted": accepted,
        "weights": str(weights),
    }
    if negatives is not None:
        entry["negatives"] = negatives
    if accepted:
        exported = yolo(str(weights)).export(format="onnx", imgsz=args.image_size)
        entry["onnx"] = str(exported)
    registry.append(entry)
    registry_path.write_text(
        json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return entry


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path, default=Path("data/processed/vine/data.yaml")
    )
    parser.add_argument("--out", type=Path, default=Path("results/vine"))
    parser.add_argument("--model", default=START_MODEL)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument(
        "--negatives",
        type=Path,
        default=None,
        help="папка снимков без стволов и гроздей для проверки ложных рамок",
    )
    entry = run(parser.parse_args(argv))
    print(json.dumps(entry, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
