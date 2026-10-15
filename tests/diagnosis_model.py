"""Крошечная ONNX-модель для тестов вместо настоящих весов (их нет в git).

Логиты — средние значения каналов, умноженные на 5: красный снимок даёт
«healthy», зелёный — «downy_mildew», синий — «powdery_mildew»."""

import io
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from PIL import Image

CLASSES = [
    "healthy",
    "black_rot",
    "esca",
    "leaf_blight",
    "downy_mildew",
    "powdery_mildew",
]
IMAGE_SIZE = 32


def build_model(path: Path) -> Path:
    weights = np.zeros((3, len(CLASSES)), dtype=np.float32)
    weights[0, CLASSES.index("healthy")] = 5
    weights[1, CLASSES.index("downy_mildew")] = 5
    weights[2, CLASSES.index("powdery_mildew")] = 5
    graph = helper.make_graph(
        [
            helper.make_node("GlobalAveragePool", ["image"], ["pooled"]),
            helper.make_node("Flatten", ["pooled"], ["flat"]),
            helper.make_node("MatMul", ["flat", "weights"], ["logits"]),
        ],
        "tiny",
        [
            helper.make_tensor_value_info(
                "image", TensorProto.FLOAT, ["batch", 3, IMAGE_SIZE, IMAGE_SIZE]
            )
        ],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["batch", 6])],
        initializer=[numpy_helper.from_array(weights, "weights")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    onnx.save(model, path)
    path.with_suffix(".json").write_text(
        json.dumps({"name": "tiny", "classes": CLASSES, "image_size": IMAGE_SIZE}),
        encoding="utf-8",
    )
    return path


def photo(
    color: tuple[int, int, int],
    size: tuple[int, int] = (100, 80),
    image_format: str = "JPEG",
) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, image_format)
    return buffer.getvalue()
