"""INT8-квантизация на крошечной свёрточной сети (ml/pr-020)."""

from pathlib import Path

import numpy as np
import pytest

from vitiguard_ml.quantize import batches, compare, softmax

FloatArray = np.ndarray


def test_compare_counts_same_diagnosis_and_probability_gap() -> None:
    reference = np.array([[2.0, 0.0], [0.0, 2.0]], dtype=np.float32)
    candidate = np.array([[2.0, 0.0], [2.0, 0.0]], dtype=np.float32)

    result = compare(reference, candidate)

    assert result["same_diagnosis"] == 0.5
    assert result["max_prob_diff"] == pytest.approx(
        float(softmax(reference)[1, 1] - softmax(candidate)[1, 1]), abs=1e-4
    )
    assert compare(reference, reference)["mean_prob_diff"] == 0


def test_batches_split_paths() -> None:
    paths = [Path(str(index)) for index in range(5)]

    sizes = [
        len(batch)
        for batch in batches(paths, lambda _: np.zeros(2, np.float32), size=2)
    ]

    assert sizes == [2, 2, 1]


def tiny_classifier(path: Path) -> Path:
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper, numpy_helper

    rng = np.random.default_rng(0)
    weights = rng.normal(size=(4, 3, 3, 3)).astype(np.float32)
    dense = rng.normal(size=(4, 2)).astype(np.float32)
    graph = helper.make_graph(
        [
            helper.make_node("Conv", ["image", "w"], ["conv"], pads=[1, 1, 1, 1]),
            helper.make_node("Relu", ["conv"], ["relu"]),
            helper.make_node("GlobalAveragePool", ["relu"], ["pool"]),
            helper.make_node("Flatten", ["pool"], ["flat"]),
            helper.make_node("MatMul", ["flat", "dense"], ["logits"]),
        ],
        "tiny",
        [helper.make_tensor_value_info("image", TensorProto.FLOAT, ["n", 3, 8, 8])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["n", 2])],
        [
            numpy_helper.from_array(weights, "w"),
            numpy_helper.from_array(dense, "dense"),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    onnx.save(model, path)
    return path


def test_int8_keeps_diagnoses_of_float_model(tmp_path: Path) -> None:
    pytest.importorskip("onnxruntime.quantization")
    from vitiguard_ml.quantize import onnx_logits, quantize_model

    source = tiny_classifier(tmp_path / "tiny.onnx")
    rng = np.random.default_rng(1)
    images = {
        Path(f"img{index}"): rng.random((3, 8, 8), dtype=np.float32)
        for index in range(32)
    }
    paths = list(images)

    target = quantize_model(
        source, tmp_path / "tiny.int8.onnx", paths[:16], images.__getitem__
    )

    assert target.exists()
    assert not (tmp_path / "tiny.int8.prep.onnx").exists()
    fp32 = onnx_logits(source, batches(paths, images.__getitem__))
    int8 = onnx_logits(target, batches(paths, images.__getitem__))
    assert compare(fp32, int8)["same_diagnosis"] >= 0.9
