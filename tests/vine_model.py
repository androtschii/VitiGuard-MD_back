"""Крошечная модель лозы для тестов: выход YOLO не зависит от снимка.

Вход — квадрат 32×32. Рамки заданы в пикселях этого квадрата: ствол, его
дубль с меньшей уверенностью (отсеивается), гроздь и слабая рамка ниже порога."""

import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

SIZE = 32
# cx, cy, w, h, оценка ствола, оценка грозди
BOXES = [
    (16, 16, 8, 16, 0.9, 0.0),
    (16, 17, 8, 16, 0.8, 0.0),
    (8, 12, 4, 4, 0.0, 0.7),
    (24, 12, 4, 4, 0.0, 0.1),
]


def build_vine_model(path: Path, with_meta: bool = True) -> Path:
    output = np.array(BOXES, dtype=np.float32).T[None]
    graph = helper.make_graph(
        [
            helper.make_node("ReduceMean", ["images"], ["mean"], keepdims=0),
            helper.make_node("Mul", ["mean", "zero"], ["nothing"]),
            helper.make_node("Add", ["boxes", "nothing"], ["output0"]),
        ],
        "tiny-yolo",
        [
            helper.make_tensor_value_info(
                "images", TensorProto.FLOAT, [1, 3, SIZE, SIZE]
            )
        ],
        [helper.make_tensor_value_info("output0", TensorProto.FLOAT, [1, 6, 4])],
        initializer=[
            numpy_helper.from_array(output, "boxes"),
            numpy_helper.from_array(np.array(0, dtype=np.float32), "zero"),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    onnx.save(model, path)
    if with_meta:
        path.with_suffix(".json").write_text(
            json.dumps(
                {"name": "tiny-yolo", "classes": ["trunk", "bunch"], "image_size": SIZE}
            ),
            encoding="utf-8",
        )
    return path
