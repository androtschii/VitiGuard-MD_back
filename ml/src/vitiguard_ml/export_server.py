"""Экспорт обученных моделей для сервера (backend/pr-046, pr-047).

    python -m vitiguard_ml.export_server --weights results/models --out ../models

Классификатор экспортируется с двумя выходами: логиты и карта активации классов
(CAM). У EfficientNet-B0 после последнего слоя признаков стоят глобальное
усреднение и один линейный слой, поэтому Grad-CAM на этом слое совпадает с CAM:
градиент логита класса по карте признака k равен весу w[c, k], делённому на
площадь карты. Карту можно посчитать прямым проходом, и серверу не нужны
градиенты и PyTorch. U-Net экспортируется как есть: вход 256×256, выход — логит
поражения для каждого пикселя."""

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch import nn

from vitiguard_ml.config import load_config
from vitiguard_ml.segmentation import create_unet
from vitiguard_ml.training import create_model

SEGMENTATION_SIZE = 256


class ClassifierWithCam(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model: nn.Module = model

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Методы timm (forward_features, forward_head) не описаны в типах nn.Module
        model: Any = self.model
        features = model.forward_features(image)
        logits = model.forward_head(features)
        weight = model.get_classifier().weight
        cam = torch.einsum("nkhw,ck->nchw", features, weight)
        return logits, cam


def export_classifier(
    name: str, weights: Path, classes: list[str], image_size: int, out: Path
) -> Path:
    model = create_model(name, len(classes), pretrained=False)
    model.load_state_dict(torch.load(weights, map_location="cpu"))
    wrapper = ClassifierWithCam(model).eval()
    path = out / "diagnosis.onnx"
    torch.onnx.export(
        wrapper,
        (torch.randn(1, 3, image_size, image_size),),
        str(path),
        input_names=["image"],
        output_names=["logits", "cam"],
        dynamic_axes={
            "image": {0: "batch"},
            "logits": {0: "batch"},
            "cam": {0: "batch"},
        },
        opset_version=17,
        dynamo=False,
    )
    meta = {"name": name, "classes": classes, "image_size": image_size}
    path.with_suffix(".json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return path


def export_unet(weights: Path, out: Path) -> Path:
    model = create_unet(pretrained=False)
    model.load_state_dict(torch.load(weights, map_location="cpu"))
    model.eval()
    path = out / "lesion.onnx"
    torch.onnx.export(
        model,
        (torch.randn(1, 3, SEGMENTATION_SIZE, SEGMENTATION_SIZE),),
        str(path),
        input_names=["image"],
        output_names=["lesion"],
        dynamic_axes={"image": {0: "batch"}, "lesion": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    meta = {"name": "unet_resnet34", "image_size": SEGMENTATION_SIZE}
    path.with_suffix(".json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", default="results/models")
    parser.add_argument("--out", default="../models")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--model", default="efficientnet_b0")
    args = parser.parse_args(argv)

    config = load_config(Path(args.config))
    weights, out = Path(args.weights), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    classifier = export_classifier(
        args.model,
        weights / f"{args.model}.pt",
        list(config.classes),
        config.image_size,
        out,
    )
    print(f"Классификатор с картой активации: {classifier}")
    print(f"Сегментация поражения: {export_unet(weights / 'unet.pt', out)}")


if __name__ == "__main__":
    main()
