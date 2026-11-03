"""Подготовка открытого набора Mendeley Data «Vineyard dataset for pruning» (DOI 10.17632/n8cs4ns97p.2).

536 полевых снимков кустов винограда в разные вегетативные фазы (Бадахос, Испания).
Разметка в формате VGG Image Annotator (VIA JSON): полигоны стволов («vine trunk»)
и побегов до/после обрезки.

Модуль:
- парсит полигоны VIA JSON;
- аппроксимирует их в ограничивающие рамки YOLO (cx, cy, w, h);
- конвертирует метку «vine trunk» в класс 0 («trunk»);
- разбивает на обучение и проверку (в тест не идут, тестовая выборка фиксирована);
- отсеивает возможные копии кадров по dHash.
"""

import contextlib
import json
import shutil
from pathlib import Path

from vitiguard_ml.data.images import IMAGE_SUFFIXES, difference_hash

TRUNK_LABELS = {"vine trunk", "trunk", "stem"}


def _hamming(a: str, b: str) -> int:
    return bin(int(a, 16) ^ int(b, 16)).count("1")


def polygon_to_bbox(
    points_x: list[float], points_y: list[float], width: float, height: float
) -> tuple[float, float, float, float] | None:
    """Конвертирует список вершин полигона VIA в нормализованные (cx, cy, w, h) YOLO."""
    if not points_x or not points_y or len(points_x) != len(points_y):
        return None
    if width <= 0 or height <= 0:
        return None

    x_min = max(0.0, min(points_x))
    x_max = min(width, max(points_x))
    y_min = max(0.0, min(points_y))
    y_max = min(height, max(points_y))

    w = x_max - x_min
    h = y_max - y_min

    if w < 1.0 or h < 1.0:
        return None

    cx = (x_min + w / 2.0) / width
    cy = (y_min + h / 2.0) / height
    norm_w = w / width
    norm_h = h / height

    return (
        round(float(cx), 6),
        round(float(cy), 6),
        round(float(norm_w), 6),
        round(float(norm_h), 6),
    )


def parse_via_json(
    annotations_json: Path, images_dir: Path
) -> dict[str, list[tuple[float, float, float, float]]]:
    """Парсит файл разметки VIA JSON и возвращает имя файла -> список рамок стволов."""
    with annotations_json.open(encoding="utf-8") as f:
        data = json.load(f)

    results: dict[str, list[tuple[float, float, float, float]]] = {}
    # В VIA структура может быть словарём записей по ключам или под ключом _via_img_metadata
    entries = data.get("_via_img_metadata", data)

    from PIL import Image

    for item in entries.values():
        if not isinstance(item, dict):
            continue
        filename = item.get("filename")
        if not filename:
            continue

        img_path = images_dir / filename
        if not img_path.exists():
            continue

        try:
            with Image.open(img_path) as im:
                w_img, h_img = float(im.width), float(im.height)
        except Exception:
            continue

        boxes: list[tuple[float, float, float, float]] = []
        regions = item.get("regions", [])
        for reg in regions:
            reg_attr = reg.get("region_attributes", {})
            label = (
                str(
                    reg_attr.get("label")
                    or reg_attr.get("name")
                    or reg_attr.get("class", "")
                )
                .strip()
                .lower()
            )
            if label not in TRUNK_LABELS:
                continue

            shape = reg.get("shape_attributes", {})
            name = shape.get("name")
            if name == "polygon":
                px = shape.get("all_points_x", [])
                py = shape.get("all_points_y", [])
                bbox = polygon_to_bbox(px, py, w_img, h_img)
                if bbox is not None:
                    boxes.append(bbox)
            elif name == "rect":
                rx = float(shape.get("x", 0))
                ry = float(shape.get("y", 0))
                rw = float(shape.get("width", 0))
                rh = float(shape.get("height", 0))
                px = [rx, rx + rw, rx + rw, rx]
                py = [ry, ry, ry + rh, ry + rh]
                bbox = polygon_to_bbox(px, py, w_img, h_img)
                if bbox is not None:
                    boxes.append(bbox)

        if boxes:
            results[filename] = boxes

    return results


def prepare_mendeley(
    raw_dir: Path,
    out: Path,
    val_ratio: float = 0.20,
    max_hamming: int = 6,
) -> dict[str, int]:
    """Разбивает размеченные кадры Mendeley на train и val (без test)."""
    images_dir = raw_dir / "mendeley" / "images"
    ann_file = raw_dir / "mendeley" / "via_annotations.json"

    if not ann_file.exists() or not images_dir.exists():
        return {"train": 0, "val": 0, "test": 0}

    parsed = parse_via_json(ann_file, images_dir)
    sorted_files = sorted(parsed.keys())

    # Сбор хэшей тестовых кадров (если уже есть test)
    from PIL import Image

    test_hashes: set[str] = set()
    test_dir = out / "images" / "test"
    if test_dir.exists():
        for p in test_dir.iterdir():
            if p.suffix.lower() in IMAGE_SUFFIXES:
                with contextlib.suppress(Exception):
                    test_hashes.add(difference_hash(Image.open(p).convert("RGB")))

    counts = {"train": 0, "val": 0, "test": 0}
    val_step = round(1.0 / val_ratio) if val_ratio > 0 else 999999

    for index, filename in enumerate(sorted_files):
        img_src = images_dir / filename
        boxes = parsed[filename]

        if test_hashes:
            with contextlib.suppress(Exception):
                h = difference_hash(Image.open(img_src).convert("RGB"))
                if any(_hamming(h, ref) <= max_hamming for ref in test_hashes):
                    continue

        split = "val" if (index % val_step == val_step - 1) else "train"

        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

        target_name = f"mendeley_{Path(filename).stem}"
        shutil.copyfile(img_src, out / "images" / split / f"{target_name}.jpg")

        yolo_lines = [f"0 {cx} {cy} {w} {h}" for cx, cy, w, h in boxes]
        (out / "labels" / split / f"{target_name}.txt").write_text(
            "\n".join(yolo_lines)
        )
        counts[split] += 1

    return counts
