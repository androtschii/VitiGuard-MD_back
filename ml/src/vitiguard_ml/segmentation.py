"""Сегментация поражения: U-Net учится на автоматических масках OpenCV
(vitiguard_ml.masks) и даёт степень поражения листа.

segmentation-models-pytorch и OpenCV входят в группу зависимостей train."""

from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import segmentation_models_pytorch as smp
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

from vitiguard_ml.augment import IMAGENET_MEAN, IMAGENET_STD
from vitiguard_ml.masks import lesion_masks
from vitiguard_ml.metrics import dice, iou


class LesionDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Снимок и его маска поражения. Маски считаются один раз при создании:
    правила OpenCV детерминированы, пересчитывать их каждую эпоху незачем."""

    def __init__(self, paths: Sequence[Path], size: int, flip: bool = False) -> None:
        self.size = size
        self.flip = flip
        self.images: list[np.ndarray] = []
        self.masks: list[np.ndarray] = []
        for path in paths:
            image = np.asarray(
                Image.open(path)
                .convert("RGB")
                .resize((size, size), Image.Resampling.BILINEAR)
            )
            self.images.append(image)
            self.masks.append(lesion_masks(image).lesion.astype(np.float32))

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, position: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = self.images[position].astype(np.float32) / 255
        mask = self.masks[position]
        if self.flip and np.random.rand() < 0.5:
            image, mask = image[:, ::-1], mask[:, ::-1]
        image = (image - np.array(IMAGENET_MEAN)) / np.array(IMAGENET_STD)
        return (
            torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float(),
            torch.from_numpy(np.ascontiguousarray(mask))[None],
        )


def create_unet(pretrained: bool = True) -> nn.Module:
    model: nn.Module = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet" if pretrained else None,
        classes=1,
    )
    return model


def train_unet(
    model: nn.Module,
    train_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    epochs: int,
    device: str,
    log: Callable[[str], None] = print,
) -> list[float]:
    """BCE + Dice: Dice не даёт сети «выиграть», предсказывая пустую маску на
    листьях, где поражение занимает малую долю площади."""
    model.to(device)
    bce = nn.BCEWithLogitsLoss()
    dice_loss = smp.losses.DiceLoss(mode="binary")
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    losses = []
    for epoch in range(1, epochs + 1):
        model.train()
        total, seen = 0.0, 0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = bce(logits, masks) + dice_loss(logits, masks)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(images)
            seen += len(images)
        losses.append(round(total / max(seen, 1), 4))
        log(f"U-Net, эпоха {epoch}: loss {losses[-1]}")
    return losses


@torch.no_grad()
def evaluate_unet(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    device: str,
) -> dict[str, float]:
    """Средние IoU и Dice по снимкам относительно автоматических масок."""
    model.eval()
    ious, dices = [], []
    for images, masks in loader:
        predicted = torch.sigmoid(model(images.to(device))).cpu().numpy() > 0.5
        targets = masks.numpy() > 0.5
        for prediction, target in zip(predicted, targets, strict=True):
            ious.append(iou(prediction[0], target[0]))
            dices.append(dice(prediction[0], target[0]))
    return {
        "iou": round(float(np.mean(ious)), 4),
        "dice": round(float(np.mean(dices)), 4),
    }
