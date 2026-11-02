"""Тепловые карты моделей (ml/pr-011) на маленьких сетях без весов."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
timm = pytest.importorskip("timm")
pytest.importorskip("pytorch_grad_cam")

from vitiguard_ml.explain import (  # noqa: E402
    attention_rollout,
    gradcam_overlays,
    target_layers,
)


def test_rollout_covers_image_and_restores_fast_attention() -> None:
    model = timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=6)
    images = torch.randn(2, 3, 224, 224)

    maps = attention_rollout(model.eval(), images)

    assert maps.shape == (2, 224, 224)
    assert maps.min() >= 0
    assert maps.max() <= 1
    assert np.ptp(maps[0]) > 0  # карта не пустая, в отличие от Grad-CAM по слою
    assert all(block.attn.fused_attn for block in model.blocks)


def test_efficientnet_map_from_activated_output() -> None:
    model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=6)

    assert target_layers("efficientnet_b0", model) == [model.bn2]


def test_overlays_for_cnn_and_vit() -> None:
    images = torch.randn(1, 3, 224, 224)
    # Маленькие сети с тем же устройством: у resnet18 тоже есть layer4
    for name, timm_name in (
        ("resnet50", "resnet18"),
        ("vit_b16", "vit_tiny_patch16_224"),
    ):
        model = timm.create_model(timm_name, pretrained=False, num_classes=6)
        overlays = gradcam_overlays(name, model, images, [2])
        assert overlays[0].shape == (224, 224, 3)
