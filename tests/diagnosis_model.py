"""Крошечные ONNX-модели для тестов вместо настоящих весов (их нет в git).

Классификатор: логиты — средние значения каналов, умноженные на 5: красный
снимок даёт «healthy», зелёный — «downy_mildew», синий — «powdery_mildew».
Карта активации (по желанию) — те же веса, применённые к каждому пикселю.
Сегментация: поражение там, где красный канал преобладает над зелёным."""

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


def _image_input() -> onnx.ValueInfoProto:
    return helper.make_tensor_value_info(
        "image", TensorProto.FLOAT, ["batch", 3, IMAGE_SIZE, IMAGE_SIZE]
    )


def _save(graph: onnx.GraphProto, path: Path, meta: dict[str, object]) -> Path:
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    onnx.save(model, path)
    path.with_suffix(".json").write_text(json.dumps(meta), encoding="utf-8")
    return path


def build_model(path: Path, with_cam: bool = False) -> Path:
    weights = np.zeros((3, len(CLASSES)), dtype=np.float32)
    weights[0, CLASSES.index("healthy")] = 5
    weights[1, CLASSES.index("downy_mildew")] = 5
    weights[2, CLASSES.index("powdery_mildew")] = 5
    nodes = [
        helper.make_node("GlobalAveragePool", ["image"], ["pooled"]),
        helper.make_node("Flatten", ["pooled"], ["flat"]),
        helper.make_node("MatMul", ["flat", "weights"], ["logits"]),
    ]
    outputs = [helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["batch", 6])]
    initializers = [numpy_helper.from_array(weights, "weights")]
    if with_cam:
        kernel = weights.T.reshape(len(CLASSES), 3, 1, 1)
        nodes.append(helper.make_node("Conv", ["image", "kernel"], ["cam"]))
        outputs.append(
            helper.make_tensor_value_info(
                "cam", TensorProto.FLOAT, ["batch", 6, IMAGE_SIZE, IMAGE_SIZE]
            )
        )
        initializers.append(numpy_helper.from_array(kernel, "kernel"))
    graph = helper.make_graph(
        nodes, "tiny", [_image_input()], outputs, initializer=initializers
    )
    return _save(
        graph, path, {"name": "tiny", "classes": CLASSES, "image_size": IMAGE_SIZE}
    )


def build_segmenter(path: Path) -> Path:
    kernel = np.array([5, -5, 0], dtype=np.float32).reshape(1, 3, 1, 1)
    graph = helper.make_graph(
        [helper.make_node("Conv", ["image", "kernel"], ["lesion"])],
        "tiny-unet",
        [_image_input()],
        [
            helper.make_tensor_value_info(
                "lesion", TensorProto.FLOAT, ["batch", 1, IMAGE_SIZE, IMAGE_SIZE]
            )
        ],
        initializer=[numpy_helper.from_array(kernel, "kernel")],
    )
    return _save(graph, path, {"name": "tiny-unet", "image_size": IMAGE_SIZE})


def photo(
    color: tuple[int, int, int],
    size: tuple[int, int] = (100, 80),
    image_format: str = "JPEG",
) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, image_format)
    return buffer.getvalue()


def half_red_half_green(size: int = 64) -> Image.Image:
    image = Image.new("RGB", (size, size), (0, 200, 0))
    image.paste((220, 0, 0), (0, 0, size // 2, size))
    return image
