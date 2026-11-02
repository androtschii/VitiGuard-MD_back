"""Подготовка Embrapa WGISD для зрения робота-сборщика (ml/pr-022).

Embrapa Wine Grape Instance Segmentation Dataset: около 300 полевых снимков
гроздей пяти винных сортов, рамки гроздей в формате YOLO на всех снимках и
маски экземпляров на части из них. Лицензия CC BY-NC 4.0: для учебной и
исследовательской работы, не для продажи. DOI 10.5281/zenodo.3361736

Набор раскладывается в формат Ultralytics YOLO: detect/ — рамки гроздей,
segment/ — контуры гроздей. Тестовая выборка — авторская, из обучающей
отделяется проверочная.

Запуск из каталога ml:  uv run python -m vitiguard_ml.data.wgisd
"""

import argparse
import shutil
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

DOWNLOAD_URL = "https://github.com/thsant/wgisd/archive/refs/heads/master.zip"
INNER_ROOT = "wgisd-master"
# Сорт — в начале имени файла; он нужен для разбиения и разбора ошибок
VARIETIES = {
    "CDY": "chardonnay",
    "CFR": "cabernet_franc",
    "CSV": "cabernet_sauvignon",
    "SVB": "sauvignon_blanc",
    "SYH": "syrah",
}
CLASS_NAME = "bunch"
# Каждый шестой снимок обучающей выборки каждого сорта — в проверочную
VAL_EVERY = 6


def download(raw_dir: Path) -> Path:
    """Скачивает и распаковывает архив; повторный запуск использует скачанное."""
    archive = raw_dir / "wgisd.zip"
    if not archive.exists():
        raw_dir.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(DOWNLOAD_URL, archive)
    extracted = raw_dir / "wgisd"
    if not extracted.exists():
        with zipfile.ZipFile(archive) as zip_file:
            zip_file.extractall(extracted)
    return extracted / INNER_ROOT


def variety(name: str) -> str:
    return VARIETIES.get(name.split("_", 1)[0], "unknown")


def read_list(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def split_names(train: list[str], test: list[str]) -> dict[str, list[str]]:
    """Проверочная выборка — из обучающей, поровну по сортам и без случайности:
    повторный запуск даёт то же разбиение."""
    by_variety: dict[str, list[str]] = {}
    for name in sorted(train):
        by_variety.setdefault(variety(name), []).append(name)
    val = [
        name
        for names in by_variety.values()
        for index, name in enumerate(names)
        if index % VAL_EVERY == VAL_EVERY - 1
    ]
    return {
        "train": [name for name in train if name not in set(val)],
        "val": val,
        "test": list(test),
    }


def mask_polygon(mask: np.ndarray, tolerance: float = 0.002) -> list[float]:
    """Контур маски одной грозди в нормированных координатах YOLO: самый
    крупный связный кусок, упрощённый с точностью 0,2 % периметра."""
    import cv2

    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return []
    contour = max(contours, key=cv2.contourArea)
    contour = cv2.approxPolyDP(contour, tolerance * cv2.arcLength(contour, True), True)
    if len(contour) < 3:
        return []
    height, width = mask.shape
    points = contour.reshape(-1, 2).astype(float)
    points[:, 0] /= width
    points[:, 1] /= height
    return [round(float(value), 6) for value in points.flatten()]


def write_yaml(path: Path, root: Path) -> None:
    path.write_text(
        f"path: {root.resolve().as_posix()}\n"
        "train: images/train\nval: images/val\ntest: images/test\n"
        f"names:\n  0: {CLASS_NAME}\n"
    )


def convert_detect(source: Path, out: Path, splits: dict[str, list[str]]) -> int:
    """Рамки WGISD уже в формате YOLO: снимки и подписи только раскладываются."""
    count = 0
    for split, names in splits.items():
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        for name in names:
            shutil.copyfile(
                source / f"{name}.jpg", out / "images" / split / f"{name}.jpg"
            )
            shutil.copyfile(
                source / f"{name}.txt", out / "labels" / split / f"{name}.txt"
            )
            count += 1
    write_yaml(out / "data.yaml", out)
    return count


def convert_segment(source: Path, out: Path, splits: dict[str, list[str]]) -> int:
    """Маски экземпляров (npz, по слою на гроздь) — в контуры YOLO-seg."""
    count = 0
    for split, names in splits.items():
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        for name in names:
            masks = np.load(source / f"{name}.npz")["arr_0"]
            lines = []
            for index in range(masks.shape[-1]):
                polygon = mask_polygon(masks[..., index])
                if polygon:
                    lines.append(" ".join(["0", *map(str, polygon)]))
            shutil.copyfile(
                source / f"{name}.jpg", out / "images" / split / f"{name}.jpg"
            )
            (out / "labels" / split / f"{name}.txt").write_text("\n".join(lines))
            count += 1
    write_yaml(out / "data.yaml", out)
    return count


def prepare(root: Path, out: Path) -> dict[str, int]:
    source = root / "data"
    detect = split_names(read_list(root / "train.txt"), read_list(root / "test.txt"))
    masked = split_names(
        read_list(root / "train_masked.txt"), read_list(root / "test_masked.txt")
    )
    return {
        "detect": convert_detect(source, out / "detect", detect),
        "segment": convert_segment(source, out / "segment", masked),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=Path("data/raw"))
    parser.add_argument("--out", type=Path, default=Path("data/processed/wgisd"))
    args = parser.parse_args()

    counts = prepare(download(args.raw), args.out)
    print(f"Снимков: рамки {counts['detect']}, контуры {counts['segment']}")


if __name__ == "__main__":
    main()
