"""Объяснимость: тепловые карты Grad-CAM — какие области листа повлияли на
решение модели. Агроном по ним видит, смотрит ли сеть на пятна, а не на фон.

Библиотека grad-cam входит в группу зависимостей train."""

from typing import Any

import numpy as np
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torch import nn

from vitiguard_ml.augment import IMAGENET_MEAN, IMAGENET_STD


def _vit_reshape(tensor: torch.Tensor) -> torch.Tensor:
    # Токены ViT (без служебного [CLS]) раскладываются обратно в сетку 14×14
    tokens = tensor[:, 1:, :]
    side = int(tokens.shape[1] ** 0.5)
    grid = tokens.reshape(tensor.size(0), side, side, tensor.size(2))
    return grid.permute(0, 3, 1, 2)


def target_layers(name: str, model: nn.Module) -> tuple[list[nn.Module], Any]:
    """Слой, по градиентам которого строится карта: последний свёрточный блок
    CNN или нормализация последнего блока трансформера."""
    timm_model: Any = model
    if name == "resnet50":
        return [timm_model.layer4[-1]], None
    if name == "efficientnet_b0":
        return [timm_model.conv_head], None
    return [timm_model.blocks[-1].norm1], _vit_reshape


def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """Нормализованный тензор CHW → изображение HWC в 0..1 для наложения карты."""
    image = tensor.permute(1, 2, 0).cpu().numpy()
    image = image * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN)
    clipped: np.ndarray = np.clip(image, 0, 1).astype(np.float32)
    return clipped


def gradcam_overlays(
    name: str, model: nn.Module, images: torch.Tensor, labels: list[int]
) -> list[np.ndarray]:
    """Наложения тепловой карты для истинного класса каждого снимка."""
    model.eval()
    layers, reshape = target_layers(name, model)
    with GradCAM(model=model, target_layers=layers, reshape_transform=reshape) as cam:
        maps = cam(
            input_tensor=images,
            targets=[ClassifierOutputTarget(label) for label in labels],
        )
    return [
        show_cam_on_image(denormalize(image), heat, use_rgb=True)
        for image, heat in zip(images, maps, strict=True)
    ]
