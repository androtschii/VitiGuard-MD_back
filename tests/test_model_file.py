import hashlib
import json
from pathlib import Path

from app.ml.model_file import describe_model, describe_robot_model


def test_version_follows_file_contents(tmp_path: Path) -> None:
    model = tmp_path / "leaf.onnx"
    model.write_bytes(b"first")
    model.with_suffix(".json").write_text(
        json.dumps({"name": "leaf", "classes": ["healthy"], "image_size": 8}),
        encoding="utf-8",
    )

    first = describe_model(model)
    # Повторный вызов без изменений файла берёт описание из кэша
    assert describe_model(model) is first
    assert first.version == hashlib.sha256(b"first").hexdigest()[:16]
    assert first.size_bytes == 5

    model.write_bytes(b"second version")

    second = describe_model(model)
    assert second.version == hashlib.sha256(b"second version").hexdigest()[:16]
    assert second.size_bytes == 14


def test_describe_robot_model(tmp_path: Path) -> None:
    vine = tmp_path / "vine.onnx"
    vine.write_bytes(b"yolo-vine-detector-bytes")

    info = describe_robot_model(vine)
    assert info.name == "vine_yolo"
    assert info.classes == ["trunk", "bunch"]
    assert info.image_size == 640
    assert info.size_bytes == len(b"yolo-vine-detector-bytes")
    assert info.version == hashlib.sha256(b"yolo-vine-detector-bytes").hexdigest()[:16]
