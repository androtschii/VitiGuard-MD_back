import json
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("timm")

from vitiguard_ml import export_server  # noqa: E402
from vitiguard_ml.training import create_model  # noqa: E402

CLASSES = [
    "healthy",
    "black_rot",
    "esca",
    "leaf_blight",
    "downy_mildew",
    "powdery_mildew",
]


def test_cam_averages_to_logits() -> None:
    # Свойство, на котором держится серверная карта: среднее CAM класса по
    # пространству плюс смещение классификатора — это его логит
    model = create_model("efficientnet_b0", len(CLASSES), pretrained=False).eval()
    wrapper = export_server.ClassifierWithCam(model).eval()
    image = torch.randn(2, 3, 224, 224)

    with torch.no_grad():
        logits, cam = wrapper(image)

    assert cam.shape == (2, len(CLASSES), 7, 7)
    head: Any = model
    bias = head.get_classifier().bias
    torch.testing.assert_close(
        cam.mean(dim=(2, 3)) + bias, logits, rtol=1e-4, atol=1e-4
    )


def test_export_writes_model_and_description(tmp_path: Path) -> None:
    model = create_model("efficientnet_b0", len(CLASSES), pretrained=False)
    weights = tmp_path / "efficientnet_b0.pt"
    torch.save(model.state_dict(), weights)

    path = export_server.export_classifier(
        "efficientnet_b0", weights, CLASSES, 224, tmp_path
    )

    assert path.stat().st_size > 1_000_000
    meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    assert meta == {"name": "efficientnet_b0", "classes": CLASSES, "image_size": 224}
