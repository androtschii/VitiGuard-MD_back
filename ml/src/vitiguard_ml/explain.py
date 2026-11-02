"""Объяснимость (ml/pr-011): какие области листа повлияли на решение модели.
Агроном по карте видит, смотрит ли сеть на пятна, а не на фон.

Для CNN — Grad-CAM, для ViT — attention rollout: у трансформера карта
градиентов по одному слою получалась пустой, а внимание всех слоёв вместе
показывает, на какие участки снимка опирается служебный токен класса.

Библиотека grad-cam входит в группу зависимостей train."""

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torch import nn

from vitiguard_ml.augment import IMAGENET_MEAN, IMAGENET_STD


def target_layers(name: str, model: nn.Module) -> list[nn.Module]:
    """Слой, по градиентам которого строится карта. У EfficientNet — выход
    последнего блока после активации, а не свёртка до неё: по свёртке карта
    выделяла фон, а сервер строит карту именно по активированному выходу."""
    timm_model: Any = model
    if name == "resnet50":
        return [timm_model.layer4[-1]]
    return [timm_model.bn2]


def attention_rollout(model: nn.Module, images: torch.Tensor) -> np.ndarray:
    """Карты внимания ViT размером со снимок, от 0 до 1. Внимание каждого слоя
    усредняется по головам, к нему добавляется остаточная связь, и слои
    перемножаются (Abnar, Zuidema, 2020); берётся строка токена класса."""
    timm_model: Any = model
    captured: list[torch.Tensor] = []
    hooks = []
    for block in timm_model.blocks:
        # Быстрое внимание PyTorch не отдаёт матрицу весов: считаем явно
        block.attn.fused_attn = False
        hooks.append(
            block.attn.attn_drop.register_forward_hook(
                lambda _module, _inputs, output: captured.append(output.detach())
            )
        )
    try:
        with torch.no_grad():
            model(images)
    finally:
        for hook in hooks:
            hook.remove()
        for block in timm_model.blocks:
            block.attn.fused_attn = True

    tokens = captured[0].shape[-1]
    identity = torch.eye(tokens, device=images.device)
    rollout = identity.expand(images.size(0), tokens, tokens)
    for attention in captured:
        layer = attention.mean(dim=1) + identity
        layer = layer / layer.sum(dim=-1, keepdim=True)
        rollout = layer @ rollout
    # Внимание токена класса к фрагментам снимка — в сетку фрагментов
    weights = rollout[:, 0, 1:]
    side = int(weights.shape[-1] ** 0.5)
    grid = weights.reshape(-1, 1, side, side)
    maps = F.interpolate(grid, size=images.shape[-2:], mode="bilinear")[:, 0]
    low = maps.amin(dim=(1, 2), keepdim=True)
    high = maps.amax(dim=(1, 2), keepdim=True)
    scaled: np.ndarray = ((maps - low) / (high - low + 1e-8)).cpu().numpy()
    return scaled


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
    if name.startswith("vit"):
        maps = attention_rollout(model, images)
    else:
        with GradCAM(model=model, target_layers=target_layers(name, model)) as cam:
            maps = cam(
                input_tensor=images,
                targets=[ClassifierOutputTarget(label) for label in labels],
            )
    return [
        show_cam_on_image(denormalize(image), heat, use_rgb=True)
        for image, heat in zip(images, maps, strict=True)
    ]
