import io
from pathlib import Path

import pytest
from diagnosis_model import CLASSES, build_model, photo
from PIL import Image

from app.core.errors import BadRequestError, UnsupportedMediaError
from app.ml.classifier import LeafClassifier
from app.services.diagnosis import read_leaf_image

RED = (255, 0, 0)


@pytest.fixture
def classifier(tmp_path: Path) -> LeafClassifier:
    return LeafClassifier(build_model(tmp_path / "tiny.onnx"))


def test_classifier_reads_metadata(classifier: LeafClassifier) -> None:
    assert classifier.name == "tiny"
    assert classifier.classes == CLASSES
    assert classifier.preprocess(Image.new("RGB", (100, 80))).shape == (1, 3, 32, 32)


def test_predictions_are_sorted_probabilities(classifier: LeafClassifier) -> None:
    predictions = classifier.predict(Image.new("RGB", (100, 80), RED))

    assert predictions[0].disease == "healthy"
    assert predictions[0].probability > 0.99
    assert {p.disease for p in predictions} == set(CLASSES)
    assert sum(p.probability for p in predictions) == pytest.approx(1)
    probabilities = [p.probability for p in predictions]
    assert probabilities == sorted(probabilities, reverse=True)


def test_grayscale_and_alpha_images_are_converted(classifier: LeafClassifier) -> None:
    for mode in ("L", "RGBA"):
        assert len(classifier.predict(Image.new(mode, (64, 64)))) == len(CLASSES)


@pytest.mark.parametrize("image_format", ["JPEG", "PNG", "WEBP"])
def test_accepts_photo_formats(image_format: str) -> None:
    image = read_leaf_image(photo(RED, image_format=image_format))

    assert image.size == (100, 80)


def test_rotates_phone_photo_by_exif() -> None:
    exif = Image.Exif()
    exif[0x0112] = 6  # снято с поворотом на 90°
    buffer = io.BytesIO()
    Image.new("RGB", (100, 80), RED).save(buffer, "JPEG", exif=exif)

    assert read_leaf_image(buffer.getvalue()).size == (80, 100)


@pytest.mark.parametrize(
    "data",
    [b"%PDF-1.7 not an image", photo(RED, image_format="GIF")],
    ids=["pdf", "gif"],
)
def test_rejects_other_formats(data: bytes) -> None:
    with pytest.raises(UnsupportedMediaError):
        read_leaf_image(data)


def test_rejects_tiny_photo() -> None:
    with pytest.raises(BadRequestError, match="слишком мал"):
        read_leaf_image(photo(RED, size=(100, 50)))


def test_rejects_decompression_bomb() -> None:
    # 48 Мпикс одного цвета сжимаются в PNG до нескольких килобайт
    buffer = io.BytesIO()
    Image.new("1", (8000, 6000)).save(buffer, "PNG")

    with pytest.raises(BadRequestError, match="разрешение"):
        read_leaf_image(buffer.getvalue())


def test_rejects_truncated_file() -> None:
    noise = Image.effect_noise((200, 200), 64).convert("RGB")
    buffer = io.BytesIO()
    noise.save(buffer, "JPEG")

    with pytest.raises(BadRequestError, match="повреждён"):
        read_leaf_image(buffer.getvalue()[:2000])
