"""Эксперименты модуля диагностики (ml/pr-006 — ml/pr-013) одной командой.

    python -m vitiguard_ml.experiments --data data/processed --out results

Этапы: classify — обучение и сравнение ResNet-50, EfficientNet-B0 и ViT-B/16;
explain — Grad-CAM; segment — маски OpenCV и U-Net; export — лучшая модель в
ONNX. Результаты (results.json, рисунки, веса) складываются в --out, поэтому
этапы можно запускать по отдельности и перезапускать после обрыва сессии."""

import argparse
import json
import random
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from vitiguard_ml.augment import build_eval_transform, build_train_transform
from vitiguard_ml.config import load_config
from vitiguard_ml.metrics import classification_report
from vitiguard_ml.seed import set_seed
from vitiguard_ml.training import (
    MODELS,
    Sample,
    create_model,
    export_onnx,
    latency_ms,
    make_loader,
    parameter_count,
    predict,
    read_samples,
    train,
)

TITLES = {
    "healthy": "Здоровый",
    "black_rot": "Чёрная гниль",
    "esca": "Эска",
    "leaf_blight": "Пятнистость",
    "downy_mildew": "Милдью",
    "powdery_mildew": "Оидиум",
}


def _results_path(out: Path) -> Path:
    return out / "results.json"


def load_results(out: Path) -> dict[str, Any]:
    path = _results_path(out)
    loaded: dict[str, Any] = (
        json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    )
    return loaded


def save_results(out: Path, results: dict[str, Any]) -> None:
    _results_path(out).write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def plot_confusion(report: dict[str, Any], title: str, path: Path) -> None:
    matrix = np.array(report["confusion_matrix"])
    labels = [TITLES.get(name, name) for name in report["classes"]]
    # Доли по строкам: видно, куда уходят примеры каждого класса
    shares = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    ax.imshow(shares, cmap="Purples", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=40, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Предсказанный класс")
    ax.set_ylabel("Истинный класс")
    ax.set_title(title)
    for row in range(len(labels)):
        for col in range(len(labels)):
            if matrix[row, col]:
                color = "white" if shares[row, col] > 0.6 else "#222222"
                ax.text(
                    col, row, matrix[row, col], ha="center", va="center", color=color
                )
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_curves(histories: dict[str, list[dict[str, float]]], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for name, history in histories.items():
        epochs = [entry["epoch"] for entry in history]
        axes[0].plot(epochs, [entry["train_loss"] for entry in history], label=name)
        axes[1].plot(epochs, [entry["val_macro_f1"] for entry in history], label=name)
    axes[0].set_title("Потери на обучающей выборке")
    axes[1].set_title("Macro-F1 на проверочной выборке")
    for ax in axes:
        ax.set_xlabel("Эпоха")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def by_source(
    samples: list[Sample], truths: list[int], predictions: list[int], classes: list[str]
) -> dict[str, Any]:
    """Метрики отдельно по источнику снимков: модель могла научиться различать
    фон и камеру источника вместо болезни — тогда точность по источникам разная."""
    reports = {}
    for source in sorted({sample.source for sample in samples}):
        picked = [i for i, sample in enumerate(samples) if sample.source == source]
        reports[source] = classification_report(
            [truths[i] for i in picked], [predictions[i] for i in picked], classes
        )
    return reports


def stage_classify(args: argparse.Namespace, config: Any, device: str) -> None:
    classes = list(config.classes)
    data, out = Path(args.data), Path(args.out)
    splits = data / "splits.csv"
    train_samples = read_samples(splits, data, classes, "train")
    val_samples = read_samples(splits, data, classes, "val")
    test_samples = read_samples(splits, data, classes, "test")
    field_samples = read_samples(
        data / "hermos" / "manifest.csv", data / "hermos", classes
    )
    if args.limit:
        # Короткий прогон для проверки, что всё запускается
        rng = random.Random(0)
        train_samples, val_samples, test_samples, field_samples = (
            rng.sample(group, min(args.limit, len(group)))
            for group in (train_samples, val_samples, test_samples, field_samples)
        )
    stats = json.loads((data / "class_stats.json").read_text(encoding="utf-8"))
    weights = [stats["class_weights"][name] for name in classes]
    size = config.image_size
    eval_transform = build_eval_transform(size)
    loaders = {
        "train": make_loader(
            train_samples,
            build_train_transform(size),
            config.batch_size,
            True,
            args.workers,
        ),
        "val": make_loader(val_samples, eval_transform, 64, False, args.workers),
        "test": make_loader(test_samples, eval_transform, 64, False, args.workers),
        "field": make_loader(field_samples, eval_transform, 64, False, args.workers),
    }

    results = load_results(out)
    results.setdefault("classification", {})
    histories = {}
    for name in args.models:
        print(f"=== {name} ===")
        set_seed(config.seed)
        model = create_model(name, len(classes), pretrained=not args.no_pretrained)
        epochs = args.vit_epochs if name == "vit_b16" else args.epochs
        trained = train(
            model,
            loaders["train"],
            loaders["val"],
            classes,
            weights,
            epochs,
            args.learning_rate,
            device,
            patience=args.patience,
        )
        torch.save(model.state_dict(), out / "models" / f"{name}.pt")
        truths, predictions, _ = predict(model, loaders["test"], device)
        test = classification_report(truths, predictions, classes)
        field_truths, field_predictions, _ = predict(model, loaders["field"], device)
        field = classification_report(field_truths, field_predictions, classes)
        plot_confusion(
            test, f"{name}: тестовая выборка", out / "figures" / f"cm_{name}_test.png"
        )
        plot_confusion(
            field,
            f"{name}: полевые снимки (HERMOS)",
            out / "figures" / f"cm_{name}_field.png",
        )
        results["classification"][name] = {
            "timm_model": MODELS[name],
            "train": {
                "best_epoch": trained.best_epoch,
                "best_val_macro_f1": trained.best_val_macro_f1,
                "seconds": trained.seconds,
                "history": trained.history,
            },
            "test": test,
            "test_by_source": by_source(test_samples, truths, predictions, classes),
            "field": field,
            "parameters": parameter_count(model),
            "size_mb": round(parameter_count(model) * 4 / 2**20, 1),
            "latency_ms": {
                "cpu": latency_ms(model.cpu(), size, "cpu", runs=10),
                **(
                    {"gpu": latency_ms(model, size, "cuda")} if device == "cuda" else {}
                ),
            },
        }
        histories[name] = trained.history
        save_results(out, results)
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
    all_histories = {
        name: entry["train"]["history"]
        for name, entry in results["classification"].items()
    }
    plot_curves(all_histories, out / "figures" / "training_curves.png")


def stage_explain(args: argparse.Namespace, config: Any, device: str) -> None:
    from vitiguard_ml.explain import denormalize, gradcam_overlays

    classes = list(config.classes)
    data, out = Path(args.data), Path(args.out)
    test_samples = read_samples(data / "splits.csv", data, classes, "test")
    rng = random.Random(1)
    picked = [
        rng.choice([s for s in test_samples if s.label == index])
        for index in range(len(classes))
    ]
    loader: DataLoader[tuple[torch.Tensor, int]] = make_loader(
        picked, build_eval_transform(config.image_size), len(picked), False, 0
    )
    images, labels = next(iter(loader))
    available = [n for n in args.models if (out / "models" / f"{n}.pt").exists()]
    columns = ["Снимок", *available]
    fig, axes = plt.subplots(
        len(picked),
        len(columns),
        figsize=(2.3 * len(columns), 2.3 * len(picked)),
        squeeze=False,
    )
    for row, image in enumerate(images):
        axes[row][0].imshow(denormalize(image))
        axes[row][0].set_ylabel(TITLES[classes[int(labels[row])]], fontsize=10)
    for col, name in enumerate(available, start=1):
        model = create_model(name, len(classes), pretrained=False)
        model.load_state_dict(
            torch.load(out / "models" / f"{name}.pt", map_location="cpu")
        )
        overlays = gradcam_overlays(
            name, model.to(device), images.to(device), labels.tolist()
        )
        for row, overlay in enumerate(overlays):
            axes[row][col].imshow(overlay)
    for col, title in enumerate(columns):
        axes[0][col].set_title(title, fontsize=10)
    for ax in np.ravel(axes):
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out / "figures" / "gradcam.png", dpi=150)
    plt.close(fig)


def stage_segment(args: argparse.Namespace, config: Any, device: str) -> None:
    from vitiguard_ml.masks import lesion_masks
    from vitiguard_ml.segmentation import (
        LesionDataset,
        create_unet,
        evaluate_unet,
        train_unet,
    )

    classes = list(config.classes)
    data, out = Path(args.data), Path(args.out)
    splits = data / "splits.csv"
    # Маски строятся по больным листьям: на здоровых поражения нет по определению
    diseased = [
        s
        for split in ("train", "test")
        for s in read_samples(splits, data, classes, split)
        if classes[s.label] != "healthy"
    ]
    rng = random.Random(2)
    rng.shuffle(diseased)
    count = args.limit or args.segment_images
    train_paths = [s.path for s in diseased[:count]]
    test_paths = [s.path for s in diseased[count : count + max(count // 4, 1)]]
    size = 256
    set_seed(config.seed)
    train_set = LesionDataset(train_paths, size, flip=True)
    test_set = LesionDataset(test_paths, size)
    model = create_unet(pretrained=not args.no_pretrained)
    losses = train_unet(
        model,
        DataLoader(train_set, batch_size=16, shuffle=True),
        args.segment_epochs,
        device,
    )
    scores = evaluate_unet(model, DataLoader(test_set, batch_size=16), device)
    torch.save(model.state_dict(), out / "models" / "unet.pt")

    # Примеры: снимок, маска OpenCV, маска U-Net и степень поражения
    examples = test_set.images[:4]
    fig, axes = plt.subplots(
        len(examples), 3, figsize=(7.5, 2.5 * len(examples)), squeeze=False
    )
    model.eval()
    for row, image in enumerate(examples):
        reference = lesion_masks(image)
        tensor, _ = test_set[row]
        with torch.no_grad():
            predicted = (
                torch.sigmoid(model(tensor[None].to(device)))[0, 0].cpu().numpy() > 0.5
            )
        severity = predicted[reference.leaf].mean() if reference.leaf.any() else 0.0
        axes[row][0].imshow(image)
        axes[row][1].imshow(reference.lesion, cmap="Purples")
        axes[row][2].imshow(predicted, cmap="Purples")
        axes[row][1].set_xlabel(f"OpenCV: {reference.severity:.0%}")
        axes[row][2].set_xlabel(f"U-Net: {severity:.0%}")
    for col, title in enumerate(["Снимок", "Маска OpenCV", "Маска U-Net"]):
        axes[0][col].set_title(title, fontsize=10)
    for ax in np.ravel(axes):
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out / "figures" / "segmentation.png", dpi=150)
    plt.close(fig)

    results = load_results(out)
    results["segmentation"] = {
        "encoder": "resnet34",
        "train_images": len(train_set),
        "test_images": len(test_set),
        "epochs": args.segment_epochs,
        "losses": losses,
        "test_vs_opencv_masks": scores,
    }
    save_results(out, results)


def stage_export(args: argparse.Namespace, config: Any, device: str) -> None:
    classes = list(config.classes)
    out = Path(args.out)
    results = load_results(out)
    trained = results.get("classification", {})
    if not trained:
        raise SystemExit("Сначала этап classify")
    # Для браузера важен размер: из моделей с macro-F1 не хуже лучшей на 0,01
    # выбирается самая лёгкая
    best_f1 = max(entry["test"]["macro_f1"] for entry in trained.values())
    name = min(
        (n for n, e in trained.items() if e["test"]["macro_f1"] >= best_f1 - 0.01),
        key=lambda n: trained[n]["parameters"],
    )
    model = create_model(name, len(classes), pretrained=False)
    model.load_state_dict(torch.load(out / "models" / f"{name}.pt", map_location="cpu"))
    path = out / "models" / f"{name}.onnx"
    export_onnx(model, config.image_size, path)
    results["export"] = {
        "model": name,
        "onnx_mb": round(path.stat().st_size / 2**20, 1),
        "classes": classes,
        "image_size": config.image_size,
    }
    save_results(out, results)


STAGES = {
    "classify": stage_classify,
    "explain": stage_explain,
    "segment": stage_segment,
    "export": stage_export,
}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/processed")
    parser.add_argument("--out", default="results")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument(
        "--stages", nargs="+", default=list(STAGES), choices=list(STAGES)
    )
    parser.add_argument(
        "--models", nargs="+", default=list(MODELS), choices=list(MODELS)
    )
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--vit-epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--segment-images", type=int, default=1200)
    parser.add_argument("--segment-epochs", type=int, default=15)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--limit", type=int, default=0, help="короткий прогон на N снимках"
    )
    parser.add_argument("--no-pretrained", action="store_true")
    args = parser.parse_args(argv)

    config = load_config(Path(args.config))
    out = Path(args.out)
    for folder in ("models", "figures"):
        (out / folder).mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Устройство: {device}")
    for stage in args.stages:
        print(f"##### этап {stage}")
        STAGES[stage](args, config, device)
    print(f"Готово, результаты в {out.resolve()}")


if __name__ == "__main__":
    main()
