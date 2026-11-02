"""INT8-квантизация классификатора и сверка выходов (ml/pr-020).

Модель для телефона скачивается целиком, поэтому её размер важен: статическая
квантизация в 8 бит уменьшает файл примерно вчетверо. Калибровка — снимки
проверочной выборки, сверка — тестовая: доля совпавших диагнозов и расхождение
вероятностей у PyTorch, ONNX (float32) и ONNX (INT8).

Запуск из каталога ml, после export_server:
  uv run --group train python -m vitiguard_ml.quantize --model ../models/diagnosis.onnx
"""

import argparse
import json
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image

FloatArray = NDArray[np.float32]
Preprocess = Callable[[Path], FloatArray]


def softmax(logits: FloatArray) -> FloatArray:
    shifted = np.exp(logits - logits.max(axis=1, keepdims=True))
    result: FloatArray = (shifted / shifted.sum(axis=1, keepdims=True)).astype(
        np.float32
    )
    return result


def compare(reference: FloatArray, candidate: FloatArray) -> dict[str, float]:
    """Насколько вторая модель отвечает как первая: тот же диагноз и разница
    вероятностей. Логиты сравниваются через softmax — так их видит пользователь."""
    first, second = softmax(reference), softmax(candidate)
    return {
        "same_diagnosis": round(
            float(np.mean(first.argmax(axis=1) == second.argmax(axis=1))), 4
        ),
        "mean_prob_diff": round(float(np.abs(first - second).mean()), 5),
        "max_prob_diff": round(float(np.abs(first - second).max()), 5),
    }


def batches(
    paths: Sequence[Path], preprocess: Preprocess, size: int = 16
) -> Iterator[FloatArray]:
    for start in range(0, len(paths), size):
        yield np.stack([preprocess(path) for path in paths[start : start + size]])


def onnx_logits(model: Path, images: Iterator[FloatArray]) -> FloatArray:
    import onnxruntime as ort

    session = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
    name = session.get_inputs()[0].name
    # Первый выход — логиты; карта активации для сверки не нужна
    parts = [np.asarray(session.run(None, {name: batch})[0]) for batch in images]
    return np.concatenate(parts).astype(np.float32)


def quantize_model(
    source: Path, target: Path, calibration: Sequence[Path], preprocess: Preprocess
) -> Path:
    """Статическая квантизация QDQ: веса по каналам в int8, активации в uint8."""
    from onnxruntime.quantization import (
        CalibrationDataReader,
        QuantFormat,
        QuantType,
        quantize_static,
    )
    from onnxruntime.quantization.shape_inference import quant_pre_process

    class Reader(CalibrationDataReader):  # type: ignore[misc]
        def __init__(self) -> None:
            self.items = iter(
                {"image": batch} for batch in batches(calibration, preprocess, 1)
            )

        def get_next(self) -> dict[str, FloatArray] | None:
            return next(self.items, None)

    prepared = target.with_name(target.stem + ".prep.onnx")
    quant_pre_process(str(source), str(prepared), skip_symbolic_shape=True)
    try:
        quantize_static(
            str(prepared),
            str(target),
            Reader(),
            quant_format=QuantFormat.QDQ,
            per_channel=True,
            weight_type=QuantType.QInt8,
            activation_type=QuantType.QUInt8,
        )
    finally:
        prepared.unlink(missing_ok=True)
    return target


def image_preprocess(image_size: int) -> Preprocess:
    """Та же подготовка, что на проверке и тесте при обучении."""
    from vitiguard_ml.augment import build_eval_transform

    transform = build_eval_transform(image_size)

    def run(path: Path) -> FloatArray:
        rgb = np.asarray(Image.open(path).convert("RGB"))
        image = np.asarray(transform(image=rgb)["image"], dtype=np.float32)
        return np.ascontiguousarray(image.transpose(2, 0, 1))

    return run


def torch_logits(
    weights: Path, name: str, classes: int, images: Iterator[FloatArray]
) -> FloatArray:
    import torch

    from vitiguard_ml.training import create_model

    model = create_model(name, classes, pretrained=False)
    model.load_state_dict(torch.load(weights, map_location="cpu"))
    model.eval()
    with torch.no_grad():
        parts = [model(torch.from_numpy(batch)).numpy() for batch in images]
    return np.concatenate(parts).astype(np.float32)


def run(args: argparse.Namespace) -> dict[str, Any]:
    from vitiguard_ml.training import read_samples

    meta = json.loads(args.model.with_suffix(".json").read_text(encoding="utf-8"))
    classes, image_size = list(meta["classes"]), int(meta["image_size"])
    preprocess = image_preprocess(image_size)
    splits = args.data / "splits.csv"
    rng = np.random.default_rng(42)
    val = read_samples(splits, args.data, classes, "val")
    test = read_samples(splits, args.data, classes, "test")
    calibration = [
        val[i].path
        for i in rng.choice(len(val), min(args.calibration, len(val)), replace=False)
    ]
    checked = [
        test[i].path
        for i in rng.choice(len(test), min(args.check, len(test)), replace=False)
    ]

    target = args.model.with_name(args.model.stem + ".int8.onnx")
    quantize_model(args.model, target, calibration, preprocess)
    # Описание то же: классы и размер входа у сжатой копии не меняются
    target.with_suffix(".json").write_text(
        args.model.with_suffix(".json").read_text(encoding="utf-8"), encoding="utf-8"
    )
    fp32 = onnx_logits(args.model, batches(checked, preprocess))
    int8 = onnx_logits(target, batches(checked, preprocess))
    report: dict[str, Any] = {
        "model": meta["name"],
        "calibration_images": len(calibration),
        "checked_images": len(checked),
        "fp32_mb": round(args.model.stat().st_size / 2**20, 2),
        "int8_mb": round(target.stat().st_size / 2**20, 2),
        "int8_vs_fp32": compare(fp32, int8),
    }
    if args.weights is not None:
        reference = torch_logits(
            args.weights, meta["name"], len(classes), batches(checked, preprocess)
        )
        report["onnx_vs_torch"] = compare(reference, fp32)
        report["int8_vs_torch"] = compare(reference, int8)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("../models/diagnosis.onnx"))
    parser.add_argument("--data", type=Path, default=Path("data/processed"))
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument("--calibration", type=int, default=200)
    parser.add_argument("--check", type=int, default=600)
    parser.add_argument(
        "--report", type=Path, default=Path("results/quantization.json")
    )
    print(json.dumps(run(parser.parse_args(argv)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
