"""Обучение классификаторов болезней по фото листа: перенос обучения с ImageNet,
взвешенная функция потерь, ранняя остановка по macro-F1 на проверочной выборке.

torch и timm входят в группу зависимостей train и импортируются только здесь."""

import copy
import csv
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import timm
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

from vitiguard_ml.metrics import classification_report

# Короткое имя эксперимента → модель timm. Все три предобучены на ImageNet
MODELS = {
    "resnet50": "resnet50",
    "efficientnet_b0": "efficientnet_b0",
    "vit_b16": "vit_base_patch16_224",
}


@dataclass(frozen=True)
class Sample:
    path: Path
    label: int
    source: str


def read_samples(
    csv_path: Path, root: Path, classes: Sequence[str], split: str | None = None
) -> list[Sample]:
    """Строки манифеста (или файла разбиения) нужной выборки. Классы вне списка
    модели пропускаются: так полевой набор HERMOS читается тем же кодом."""
    index = {name: number for number, name in enumerate(classes)}
    samples = []
    with csv_path.open(encoding="utf-8") as file:
        for row in csv.DictReader(file):
            if split is not None and row.get("split") != split:
                continue
            if row["label"] not in index:
                continue
            samples.append(
                Sample(root / row["path"], index[row["label"]], row["source"])
            )
    return samples


class LeafDataset(Dataset[tuple[torch.Tensor, int]]):
    def __init__(self, samples: Sequence[Sample], transform: Any) -> None:
        self.samples = list(samples)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, position: int) -> tuple[torch.Tensor, int]:
        sample = self.samples[position]
        image = np.asarray(Image.open(sample.path).convert("RGB"))
        tensor = torch.from_numpy(self.transform(image=image)["image"])
        # Albumentations отдаёт HWC, сеть ждёт CHW
        return tensor.permute(2, 0, 1).float(), sample.label


def make_loader(
    samples: Sequence[Sample],
    transform: Any,
    batch_size: int,
    shuffle: bool,
    workers: int = 2,
) -> DataLoader[tuple[torch.Tensor, int]]:
    return DataLoader(
        LeafDataset(samples, transform),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
    )


def create_model(name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    model: nn.Module = timm.create_model(
        MODELS[name], pretrained=pretrained, num_classes=num_classes
    )
    return model


@dataclass
class TrainResult:
    best_epoch: int
    best_val_macro_f1: float
    history: list[dict[str, float]] = field(default_factory=list)
    seconds: float = 0.0


@torch.no_grad()
def predict(
    model: nn.Module, loader: DataLoader[tuple[torch.Tensor, int]], device: str
) -> tuple[list[int], list[int], np.ndarray]:
    """Истинные метки, предсказания и вероятности классов (softmax)."""
    model.eval()
    truths: list[int] = []
    predictions: list[int] = []
    probabilities: list[np.ndarray] = []
    for images, labels in loader:
        logits = model(images.to(device))
        probs = torch.softmax(logits.float(), dim=1).cpu().numpy()
        probabilities.append(probs)
        predictions.extend(probs.argmax(axis=1).tolist())
        truths.extend(labels.tolist())
    stacked = np.concatenate(probabilities) if probabilities else np.empty((0, 0))
    return truths, predictions, stacked


def train(
    model: nn.Module,
    train_loader: DataLoader[tuple[torch.Tensor, int]],
    val_loader: DataLoader[tuple[torch.Tensor, int]],
    classes: Sequence[str],
    class_weights: Sequence[float],
    epochs: int,
    learning_rate: float,
    device: str,
    patience: int = 4,
    log: Callable[[str], None] = print,
) -> TrainResult:
    """Обучение с ранней остановкой: после каждой эпохи считается macro-F1 на
    проверочной выборке; лучшие веса сохраняются, а обучение прекращается, если
    patience эпох подряд нет улучшения. В конце в модель возвращаются лучшие веса."""
    model.to(device)
    weights = torch.tensor(class_weights, dtype=torch.float32, device=device)
    # Веса классов: редкий оидиум «стоит» дороже, и модель не может набрать
    # точность, просто игнорируя его
    loss_fn = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=0.05
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)

    result = TrainResult(best_epoch=0, best_val_macro_f1=-1.0)
    best_state = copy.deepcopy(model.state_dict())
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, seen = 0.0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device, enabled=use_amp):
                loss = loss_fn(model(images), labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item() * len(labels)
            seen += len(labels)
        scheduler.step()

        truths, predictions, _ = predict(model, val_loader, device)
        report = classification_report(truths, predictions, classes)
        entry = {
            "epoch": float(epoch),
            "train_loss": round(total_loss / max(seen, 1), 4),
            "val_accuracy": report["accuracy"],
            "val_macro_f1": report["macro_f1"],
        }
        result.history.append(entry)
        log(
            f"эпоха {epoch}: loss {entry['train_loss']}, "
            f"val accuracy {entry['val_accuracy']}, val macro-F1 {entry['val_macro_f1']}"
        )
        if report["macro_f1"] > result.best_val_macro_f1:
            result.best_val_macro_f1 = report["macro_f1"]
            result.best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
        elif epoch - result.best_epoch >= patience:
            log(f"ранняя остановка: {patience} эпох без улучшения")
            break

    model.load_state_dict(best_state)
    result.seconds = round(time.perf_counter() - started, 1)
    return result


@torch.no_grad()
def latency_ms(model: nn.Module, image_size: int, device: str, runs: int = 30) -> float:
    """Среднее время классификации одного снимка (пакет из одного изображения)."""
    model.eval().to(device)
    image = torch.randn(1, 3, image_size, image_size, device=device)
    for _ in range(5):
        model(image)
    if device == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(runs):
        model(image)
    if device == "cuda":
        torch.cuda.synchronize()
    return round((time.perf_counter() - started) / runs * 1000, 2)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def export_onnx(model: nn.Module, image_size: int, path: Path) -> None:
    """Модель для ONNX Runtime: в браузере (офлайн-диагностика) и на сервере."""
    model.eval().cpu()
    dummy = torch.randn(1, 3, image_size, image_size)
    torch.onnx.export(
        model,
        (dummy,),
        str(path),
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
