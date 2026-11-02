"""Крошечная ONNX-модель риска для тестов вместо обученной LightGBM: вероятность
заражения — сигмоида от суммы осадков за двое суток минус 10 мм. Выходы как
у LightGBM после onnxmltools без ZipMap: метка и вероятности классов."""

import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from app.services.risk_features import FEATURES


def build_risk_model(path: Path, threshold: float = 0.5) -> Path:
    weights = np.zeros((len(FEATURES), 1), dtype=np.float32)
    weights[FEATURES.index("rain_2d"), 0] = 1.0
    nodes = [
        helper.make_node("MatMul", ["input", "weights"], ["rain"]),
        helper.make_node("Sub", ["rain", "ten"], ["shifted"]),
        helper.make_node("Sigmoid", ["shifted"], ["positive"]),
        helper.make_node("Sub", ["one", "positive"], ["negative"]),
        helper.make_node("Concat", ["negative", "positive"], ["probabilities"], axis=1),
        helper.make_node("ArgMax", ["probabilities"], ["label"], axis=1, keepdims=0),
    ]
    graph = helper.make_graph(
        nodes,
        "risk",
        [
            helper.make_tensor_value_info(
                "input", TensorProto.FLOAT, ["n", len(FEATURES)]
            )
        ],
        [
            helper.make_tensor_value_info("label", TensorProto.INT64, ["n"]),
            helper.make_tensor_value_info("probabilities", TensorProto.FLOAT, ["n", 2]),
        ],
        [
            numpy_helper.from_array(weights, "weights"),
            numpy_helper.from_array(np.array([10.0], dtype=np.float32), "ten"),
            numpy_helper.from_array(np.array([1.0], dtype=np.float32), "one"),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    onnx.save(model, path)
    path.with_suffix(".json").write_text(
        json.dumps(
            {"name": "lightgbm", "features": list(FEATURES), "threshold": threshold}
        ),
        encoding="utf-8",
    )
    return path
