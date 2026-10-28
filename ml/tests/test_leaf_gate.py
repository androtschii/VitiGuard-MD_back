import json
import random
from pathlib import Path

import numpy as np
import pytest

from vitiguard_ml.leaf_gate import (
    FEATURES,
    add_features_output,
    fit_gate,
    images,
    rates,
    with_threshold,
)


def clouds(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    grape = rng.normal(1.0, 0.5, size=(300, 8))
    not_grape = rng.normal(-1.0, 0.5, size=(500, 8))
    return grape, not_grape


def test_gate_separates_and_keeps_field_share() -> None:
    grape, not_grape = clouds()
    field, other = clouds(seed=1)

    gate = with_threshold(fit_gate(grape, not_grape), field, keep=0.98)
    result = rates(gate, field, other)

    # Порог подобран так, что проходит заданная доля полевых листьев
    assert result["grape_passed"] == pytest.approx(0.98, abs=0.01)
    assert result["not_grape_rejected"] > 0.99


def test_unbalanced_classes_do_not_shift_scores() -> None:
    grape, not_grape = clouds()

    gate = fit_gate(grape[:30], not_grape)

    assert np.median(gate.scores(grape)) > 0.9
    assert np.median(gate.scores(not_grape)) < 0.1


def test_gate_json_round_trip() -> None:
    grape, not_grape = clouds()
    gate = with_threshold(fit_gate(grape, not_grape), grape)

    data = json.loads(gate.to_json({"grape_passed": 0.98}))

    assert len(data["weights"]) == 8
    assert data["threshold"] == pytest.approx(gate.threshold)
    assert data["metrics"] == {"grape_passed": 0.98}


def test_images_are_sampled_from_folders(tmp_path: Path) -> None:
    for name in ("a.jpg", "b.PNG", "c.txt", "sub/d.webp"):
        path = tmp_path / name
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(b"x")

    found = images([tmp_path], limit=10, rng=random.Random(0))

    assert sorted(p.name for p in found) == ["a.jpg", "b.PNG", "d.webp"]
    assert len(images([tmp_path], limit=2, rng=random.Random(0))) == 2


def test_features_output_is_added_once(tmp_path: Path) -> None:
    onnx = pytest.importorskip("onnx")
    helper = onnx.helper
    graph = helper.make_graph(
        [
            helper.make_node("Identity", ["image"], ["pooled"]),
            helper.make_node("Identity", ["pooled"], ["logits"]),
        ],
        "tiny",
        [helper.make_tensor_value_info("image", onnx.TensorProto.FLOAT, [1, 4])],
        [helper.make_tensor_value_info("logits", onnx.TensorProto.FLOAT, [1, 4])],
    )
    path = tmp_path / "model.onnx"
    onnx.save(helper.make_model(graph), path)

    add_features_output(path, path, tensor="pooled")
    add_features_output(path, path, tensor="pooled")

    outputs = [output.name for output in onnx.load(path).graph.output]
    assert outputs == ["logits", FEATURES]
