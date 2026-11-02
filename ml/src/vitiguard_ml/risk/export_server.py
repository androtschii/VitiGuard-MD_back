"""Модель риска для сервера (pr-076): LightGBM, заражение милдью по «трём
десяткам» на следующий день, в ONNX.

Протокол тот же, что в сравнении (ml/pr-018): обучение на всех годах, кроме
последнего, ранняя остановка и порог F1 — по последнему году. Рядом с моделью
пишется описание: порядок признаков, порог и качество на проверочном году.

Запуск из каталога ml:
  uv run --group train python -m vitiguard_ml.risk.export_server --out ../models
"""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from vitiguard_ml.risk.experiment import best_threshold, metrics
from vitiguard_ml.risk.features import (
    build_features,
    feature_columns,
    load_daily,
    target_column,
)
from vitiguard_ml.risk.models import predict_boosting, train_boosting

TARGET = target_column("downy", 1)


def train_and_export(data: Path, out: Path, seed: int = 42) -> dict[str, Any]:
    from onnxmltools import convert_lightgbm
    from onnxmltools.convert.common.data_types import FloatTensorType

    frame = build_features(load_daily(data))
    columns = feature_columns(frame)
    labelled = frame[frame[TARGET].notna()]
    year = labelled["date"].dt.year
    valid_year = int(year.max())
    fit = labelled[year < valid_year]
    valid = labelled[year == valid_year]

    model = train_boosting(fit, valid, columns, TARGET, seed=seed)
    scores = predict_boosting(model, valid, columns)
    labels = valid[TARGET].astype(int).to_numpy()
    threshold = best_threshold(labels, scores)
    quality = metrics(labels, scores, threshold)

    # Без ZipMap выход — обычный массив вероятностей классов [n, 2]
    onnx_model = convert_lightgbm(
        model.model.booster_,
        initial_types=[("input", FloatTensorType([None, len(columns)]))],
        zipmap=False,
        target_opset=15,
    )
    out.mkdir(parents=True, exist_ok=True)
    path = out / "risk.onnx"
    path.write_bytes(onnx_model.SerializeToString())

    # Сверка: ONNX отвечает так же, как сама LightGBM
    import onnxruntime as ort

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    sample = valid[columns].to_numpy(dtype=np.float32)
    onnx_scores = np.asarray(session.run(None, {"input": sample})[-1])[:, 1]
    # Расхождение — на границах деревьев из-за float32: важно, что решение
    # «высокий риск» то же самое
    diff = np.abs(onnx_scores - scores)
    agreement = float(np.mean((onnx_scores >= threshold) == (scores >= threshold)))

    meta = {
        "name": "lightgbm",
        "target": TARGET,
        "features": columns,
        "threshold": threshold,
        "trained_years": [int(year.min()), valid_year - 1],
        "valid_year": valid_year,
        "valid": quality,
        "onnx_check": {
            "mean_diff": round(float(diff.mean()), 6),
            "max_diff": round(float(diff.max()), 6),
            "same_decision": agreement,
        },
        "trees": int(model.model.booster_.num_trees()),
    }
    path.with_suffix(".json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return meta


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/raw/risk/daily.csv"))
    parser.add_argument("--out", type=Path, default=Path("../models"))
    args = parser.parse_args(argv)
    meta = train_and_export(args.data, args.out)
    print(json.dumps({k: v for k, v in meta.items() if k != "features"}, indent=2))


if __name__ == "__main__":
    main()
