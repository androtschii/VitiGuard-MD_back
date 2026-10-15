import base64
import io
from pathlib import Path

import numpy as np
import pytest
from diagnosis_model import (
    CLASSES,
    build_model,
    build_segmenter,
    half_red_half_green,
    photo,
)
from PIL import Image

from app.core.errors import BadRequestError, UnsupportedMediaError
from app.ml.classifier import LeafClassifier, to_input
from app.ml.overlays import heatmap, lesion_mask, overlay_size
from app.ml.segmentation import LesionSegmenter
from app.services.diagnosis import analyze_leaf, read_leaf_image

RED = (255, 0, 0)


@pytest.fixture
def classifier(tmp_path: Path) -> LeafClassifier:
    return LeafClassifier(build_model(tmp_path / "tiny.onnx"))


@pytest.fixture
def cam_classifier(tmp_path: Path) -> LeafClassifier:
    return LeafClassifier(build_model(tmp_path / "cam.onnx", with_cam=True))


@pytest.fixture
def segmenter(tmp_path: Path) -> LesionSegmenter:
    return LesionSegmenter(build_segmenter(tmp_path / "unet.onnx"))


def pixel(image: Image.Image, xy: tuple[int, int]) -> tuple[int, ...]:
    value = image.getpixel(xy)
    assert isinstance(value, tuple)
    return value


def decode(data_url: str) -> Image.Image:
    header, payload = data_url.split(",", 1)
    assert header == "data:image/png;base64"
    return Image.open(io.BytesIO(base64.b64decode(payload)))


def test_classifier_reads_metadata(classifier: LeafClassifier) -> None:
    assert classifier.name == "tiny"
    assert classifier.classes == CLASSES
    assert not classifier.has_activation
    assert to_input(Image.new("RGB", (100, 80)), 32).shape == (1, 3, 32, 32)


def test_predictions_are_sorted_probabilities(classifier: LeafClassifier) -> None:
    result = classifier.classify(Image.new("RGB", (100, 80), RED))
    predictions = result.predictions

    assert predictions[0].disease == "healthy"
    assert predictions[0].probability > 0.99
    assert {p.disease for p in predictions} == set(CLASSES)
    assert sum(p.probability for p in predictions) == pytest.approx(1)
    probabilities = [p.probability for p in predictions]
    assert probabilities == sorted(probabilities, reverse=True)
    assert result.activation is None


def test_grayscale_and_alpha_images_are_converted(classifier: LeafClassifier) -> None:
    for mode in ("L", "RGBA"):
        result = classifier.classify(Image.new(mode, (64, 64)))
        assert len(result.predictions) == len(CLASSES)


def test_activation_of_best_class(cam_classifier: LeafClassifier) -> None:
    # Красная левая половина даёт «healthy», и карта горячая именно там
    image = Image.new("RGB", (64, 64), (128, 128, 128))
    image.paste(RED, (0, 0, 32, 64))

    result = cam_classifier.classify(image)

    assert result.predictions[0].disease == "healthy"
    assert result.activation is not None
    assert result.activation.shape == (32, 32)
    assert result.activation[:, :14].min() == pytest.approx(1)
    assert result.activation[:, 18:].max() < 0.1


def test_activation_without_positive_evidence_is_zero(
    cam_classifier: LeafClassifier,
) -> None:
    # Чёрный снимок: у победившего класса нулевые веса, карта пуста, а не NaN
    result = cam_classifier.classify(Image.new("RGB", (64, 64)))

    assert result.activation is not None
    assert not result.activation.any()


def test_segmenter_measures_lesion_share(segmenter: LesionSegmenter) -> None:
    mask = segmenter.segment(half_red_half_green())

    assert segmenter.name == "tiny-unet"
    assert mask.lesion.shape == (32, 32)
    assert mask.lesion[:, :16].all()
    assert not mask.lesion[:, 16:].any()
    assert mask.severity == pytest.approx(0.5)


def test_segmenter_without_leaf_gives_zero(segmenter: LesionSegmenter) -> None:
    assert segmenter.segment(Image.new("RGB", (64, 64), (128, 128, 128))).severity == 0


def test_overlay_keeps_photo_proportions() -> None:
    assert overlay_size(1000, 500) == (448, 224)
    assert overlay_size(300, 600, 224) == (112, 224)


def test_heatmap_is_transparent_where_activation_is_low() -> None:
    activation = np.zeros((7, 7), dtype=np.float32)
    activation[3, 3] = 1

    image = decode(heatmap(activation, (70, 70)))

    assert image.mode == "RGBA"
    assert image.size == (70, 70)
    assert pixel(image, (35, 35))[3] > 150
    assert pixel(image, (2, 2))[3] == 0


def test_lesion_mask_is_red_on_lesion_only() -> None:
    lesion = np.zeros((4, 4), dtype=bool)
    lesion[:, :2] = True

    image = decode(lesion_mask(lesion, (8, 8)))

    assert pixel(image, (1, 4)) == (220, 38, 38, 150)
    assert pixel(image, (6, 4))[3] == 0


def test_analyze_with_all_models(
    cam_classifier: LeafClassifier, segmenter: LesionSegmenter
) -> None:
    result = analyze_leaf(half_red_half_green(200), cam_classifier, segmenter, 2)

    assert len(result.top) == 2
    assert result.heatmap is not None
    assert decode(result.heatmap).size == (224, 224)
    assert result.lesion is not None
    assert result.lesion.severity == pytest.approx(0.5)
    assert decode(result.lesion.mask).size == (448, 448)


def test_analyze_without_extra_models(classifier: LeafClassifier) -> None:
    result = analyze_leaf(Image.new("RGB", (100, 80), RED), classifier, None, 3)

    assert result.disease == "healthy"
    assert result.heatmap is None
    assert result.lesion is None


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
