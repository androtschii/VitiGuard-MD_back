"""Сегментация гроздей для робота-сборщика (ml/pr-023).

YOLO-seg находит каждую гроздь и её контур: по контуру считается, где
ножка (верх грозди), сколько в ней ягод на снимке и насколько ровно она
окрашена. Обучение — на GPU в Colab, результат — ONNX для компьютера робота.

Запуск из каталога ml:
  uv run --group train python -m vitiguard_ml.robot.bunches --data data/processed/wgisd/segment/data.yaml
"""

import argparse
import json
from pathlib import Path
from typing import Any

# Маленькая модель: робот считает на месте, на Jetson или ноутбуке
DEFAULT_MODEL = "yolo11n-seg.pt"


def summarize(metrics: Any) -> dict[str, float]:
    """Метрики Ultralytics в простой словарь: точность рамок и масок."""
    box, seg = metrics.box, metrics.seg
    return {
        "box_map50": float(box.map50),
        "box_map50_95": float(box.map),
        "mask_map50": float(seg.map50),
        "mask_map50_95": float(seg.map),
        "precision": float(box.mp),
        "recall": float(box.mr),
    }


def train(
    data: Path,
    out: Path,
    model_name: str = DEFAULT_MODEL,
    epochs: int = 100,
    image_size: int = 1024,
    batch: int = 8,
    seed: int = 42,
) -> dict[str, Any]:
    from ultralytics import YOLO

    model = YOLO(model_name)
    # Снимки WGISD крупные, грозди на них мелкие: размер входа больше обычного
    model.train(
        data=str(data),
        epochs=epochs,
        imgsz=image_size,
        batch=batch,
        seed=seed,
        deterministic=True,
        project=str(out),
        name="bunches",
        exist_ok=True,
        patience=20,
    )
    test = summarize(model.val(data=str(data), split="test", imgsz=image_size))
    onnx_path = model.export(format="onnx", imgsz=image_size, simplify=True)
    result = {
        "model": model_name,
        "image_size": image_size,
        "epochs": epochs,
        "test": test,
        "onnx": str(onnx_path),
    }
    (out / "bunches.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("results/robot"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=8)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    result = train(
        args.data, args.out, args.model, args.epochs, args.image_size, args.batch
    )
    print(json.dumps(result["test"], indent=2))


if __name__ == "__main__":
    main()
