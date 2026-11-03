"""Импорт исправленной человеком разметки из Label Studio или CVAT (ml/pr-031).

Человек размечает отобранные активным обучением (ml/pr-030) кадры в Label Studio
или CVAT. Модуль выполняет:
1. Чтение разметки (Label Studio JSON или YOLO/CVAT).
2. Валидацию геометрии рамок и приведение классов к единому словарю (trunk=0, bunch=1).
3. Проверку на утечку данных (data leakage): сравнение dHash с тестовой выборкой.
   Кадры с расстоянием Хэмминга <= 5 от любого тестового кадра отбрасываются.
4. Отсеивание дубликатов с уже имеющимися обучающими кадрами.
5. Запись изображений и меток в data/processed/vine/images/train/ и labels/train/.
6. Ведение журнала пополнения обучающего набора в ingestion_log.jsonl.

Запуск из каталога ml:
  uv run python -m vitiguard_ml.robot.annotation_import \\
      --input tasks_completed.json \\
      --images data/raw/daily_frames \\
      --dataset data/processed/vine
"""

import argparse
import json
import shutil
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from vitiguard_ml.data.images import (
    IMAGE_SUFFIXES,
    hamming,
    read_image_info,
)
from vitiguard_ml.data.vine import MAX_HAMMING, collect_test_hashes

LABEL_MAP: dict[str, int] = {
    "trunk": 0,
    "vine_trunk": 0,
    "truk": 0,
    "bunch": 1,
    "grape": 1,
    "grape_bunch": 1,
    "tiny_grape_bunch": 1,
    "medium_grape_bunch": 1,
}

CLASS_NAMES = ("trunk", "bunch")


@dataclass(frozen=True)
class NormalizedBox:
    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float

    def to_yolo_line(self) -> str:
        return (
            f"{self.class_id} "
            f"{self.x_center:.6f} "
            f"{self.y_center:.6f} "
            f"{self.width:.6f} "
            f"{self.height:.6f}\n"
        )


@dataclass
class ImportSummary:
    total_tasks: int = 0
    added_images: int = 0
    skipped_duplicates: int = 0
    rejected_leaks: int = 0
    invalid_boxes: int = 0
    total_boxes: int = 0
    boxes_by_class: dict[str, int] = field(
        default_factory=lambda: {"trunk": 0, "bunch": 0}
    )
    details: list[dict[str, Any]] = field(default_factory=list)


def parse_label_studio_result(item: dict[str, Any]) -> NormalizedBox | None:
    """Извлекает нормализованную рамку YOLO из элемента разметки Label Studio."""
    result_type = item.get("type")
    val = item.get("value", {})
    if not val:
        return None

    raw_labels = val.get("rectanglelabels") or val.get("polygonlabels") or []
    if not raw_labels:
        return None

    label_str = str(raw_labels[0]).lower().strip()
    if label_str not in LABEL_MAP:
        return None
    class_id = LABEL_MAP[label_str]

    if result_type == "rectanglelabels":
        x = float(val.get("x", 0.0))
        y = float(val.get("y", 0.0))
        w = float(val.get("width", 0.0))
        h = float(val.get("height", 0.0))

        if w <= 0.0 or h <= 0.0:
            return None

        x_c = (x + w / 2.0) / 100.0
        y_c = (y + h / 2.0) / 100.0
        bw = w / 100.0
        bh = h / 100.0

    elif result_type == "polygonlabels":
        points = val.get("points", [])
        if not points or len(points) < 3:
            return None

        xs = [float(p[0]) for p in points]
        ys = [float(p[1]) for p in points]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        w = max_x - min_x
        h = max_y - min_y

        if w <= 0.0 or h <= 0.0:
            return None

        x_c = (min_x + w / 2.0) / 100.0
        y_c = (min_y + h / 2.0) / 100.0
        bw = w / 100.0
        bh = h / 100.0
    else:
        return None

    # Ограничение диапазоном [0.0, 1.0]
    x_c = max(0.0, min(1.0, x_c))
    y_c = max(0.0, min(1.0, y_c))
    bw = max(0.001, min(1.0, bw))
    bh = max(0.001, min(1.0, bh))

    return NormalizedBox(
        class_id=class_id,
        x_center=round(x_c, 6),
        y_center=round(y_c, 6),
        width=round(bw, 6),
        height=round(bh, 6),
    )


def extract_task_boxes(task: dict[str, Any]) -> list[NormalizedBox]:
    """Извлекает все валидные рамки из задачи Label Studio (из annotations или predictions)."""
    boxes: list[NormalizedBox] = []
    annotations = task.get("annotations", [])
    if not annotations:
        annotations = task.get("predictions", [])

    for ann in annotations:
        results = ann.get("result", [])
        for item in results:
            box = parse_label_studio_result(item)
            if box is not None:
                boxes.append(box)
    return boxes


def resolve_image_path(task: dict[str, Any], images_dir: Path | None) -> Path | None:
    """Определяет локальный путь к изображению по данным задачи Label Studio."""
    data = task.get("data", {})
    image_ref = (
        data.get("image") or data.get("image_url") or task.get("file_upload") or ""
    )
    if not image_ref:
        return None

    # Если в URL передан путь
    filename = Path(image_ref).name
    # Удаление возможного хэша префикса Label Studio (например, abc1234-frame.jpg)
    if (
        "-" in filename
        and images_dir is not None
        and not (images_dir / filename).exists()
    ):
        candidate_clean = filename.split("-", 1)[-1]
        if (images_dir / candidate_clean).exists():
            return images_dir / candidate_clean

    if images_dir is not None:
        direct = images_dir / filename
        if direct.exists():
            return direct
        # Рекурсивный поиск в images_dir при необходимости
        for found in images_dir.glob(f"**/{filename}"):
            return found

    direct_path = Path(image_ref)
    if direct_path.exists():
        return direct_path

    return None


def collect_train_hashes(dataset_dir: Path) -> set[str]:
    """Собирает dHash всех существующих обучающих кадров."""
    train_dir = dataset_dir / "images" / "train"
    if not train_dir.exists():
        return set()
    hashes: set[str] = set()
    for p in sorted(train_dir.iterdir()):
        if p.suffix.lower() in IMAGE_SUFFIXES:
            info = read_image_info(p)
            if info is not None:
                hashes.add(info.dhash)
    return hashes


class AnnotationImporter:
    """Импорт и валидация человеческой разметки с защитой от дубликатов и утечки теста."""

    def __init__(self, dataset_dir: Path, max_hamming: int = MAX_HAMMING) -> None:
        self.dataset_dir = dataset_dir
        self.max_hamming = max_hamming
        self.train_images_dir = dataset_dir / "images" / "train"
        self.train_labels_dir = dataset_dir / "labels" / "train"
        self.log_path = dataset_dir / "ingestion_log.jsonl"

        self.train_images_dir.mkdir(parents=True, exist_ok=True)
        self.train_labels_dir.mkdir(parents=True, exist_ok=True)

        self.test_hashes: set[str] = collect_test_hashes(dataset_dir)
        self.train_hashes: set[str] = collect_train_hashes(dataset_dir)

    def import_label_studio(
        self,
        export_path: Path,
        images_dir: Path | None = None,
    ) -> ImportSummary:
        """Импортирует задачи из файла экспорта Label Studio JSON."""
        summary = ImportSummary()
        content = json.loads(export_path.read_text(encoding="utf-8"))
        tasks: list[dict[str, Any]] = (
            content if isinstance(content, list) else [content]
        )
        summary.total_tasks = len(tasks)

        for task in tasks:
            img_path = resolve_image_path(task, images_dir)
            if img_path is None or not img_path.exists():
                summary.details.append(
                    {
                        "task_id": task.get("id"),
                        "status": "image_not_found",
                    }
                )
                continue

            info = read_image_info(img_path)
            if info is None:
                summary.details.append(
                    {
                        "task_id": task.get("id"),
                        "status": "corrupt_image",
                        "image": img_path.name,
                    }
                )
                continue

            # 1. Проверка на утечку данных в тестовую выборку
            is_leak = any(
                hamming(info.dhash, th) <= self.max_hamming for th in self.test_hashes
            )
            if is_leak:
                summary.rejected_leaks += 1
                summary.details.append(
                    {
                        "task_id": task.get("id"),
                        "status": "rejected_test_leak",
                        "image": img_path.name,
                        "dhash": info.dhash,
                    }
                )
                self._log_ingestion(
                    img_path.name,
                    info.dhash,
                    status="rejected_test_leak",
                    box_counts={},
                )
                continue

            # 2. Проверка на дубликат в обучающей выборке
            if info.dhash in self.train_hashes:
                summary.skipped_duplicates += 1
                summary.details.append(
                    {
                        "task_id": task.get("id"),
                        "status": "skipped_duplicate",
                        "image": img_path.name,
                        "dhash": info.dhash,
                    }
                )
                continue

            # 3. Извлечение рамок
            boxes = extract_task_boxes(task)
            dest_image_name = f"human_{img_path.name}"
            dest_image_path = self.train_images_dir / dest_image_name
            dest_label_path = self.train_labels_dir / f"{dest_image_path.stem}.txt"

            shutil.copyfile(img_path, dest_image_path)
            label_lines = [box.to_yolo_line() for box in boxes]
            dest_label_path.write_text("".join(label_lines), encoding="utf-8")

            # Обновление множества хэшей
            self.train_hashes.add(info.dhash)
            summary.added_images += 1
            summary.total_boxes += len(boxes)

            box_counts: dict[str, int] = {"trunk": 0, "bunch": 0}
            for box in boxes:
                cname = CLASS_NAMES[box.class_id]
                box_counts[cname] += 1
                summary.boxes_by_class[cname] += 1

            self._log_ingestion(
                dest_image_name,
                info.dhash,
                status="added",
                box_counts=box_counts,
            )

            summary.details.append(
                {
                    "task_id": task.get("id"),
                    "status": "added",
                    "image": dest_image_name,
                    "boxes": len(boxes),
                    "box_counts": box_counts,
                }
            )

        return summary

    def import_yolo_dir(
        self,
        images_src: Path,
        labels_src: Path,
    ) -> ImportSummary:
        """Импортирует кадры и метки из экспорта YOLO/CVAT."""
        summary = ImportSummary()
        image_files = [
            p
            for p in sorted(images_src.iterdir())
            if p.suffix.lower() in IMAGE_SUFFIXES
        ]
        summary.total_tasks = len(image_files)

        for img_path in image_files:
            info = read_image_info(img_path)
            if info is None:
                summary.details.append(
                    {"status": "corrupt_image", "image": img_path.name}
                )
                continue

            if any(
                hamming(info.dhash, th) <= self.max_hamming for th in self.test_hashes
            ):
                summary.rejected_leaks += 1
                summary.details.append(
                    {"status": "rejected_test_leak", "image": img_path.name}
                )
                self._log_ingestion(
                    img_path.name,
                    info.dhash,
                    status="rejected_test_leak",
                    box_counts={},
                )
                continue

            if info.dhash in self.train_hashes:
                summary.skipped_duplicates += 1
                summary.details.append(
                    {"status": "skipped_duplicate", "image": img_path.name}
                )
                continue

            label_path = labels_src / f"{img_path.stem}.txt"
            boxes: list[NormalizedBox] = []
            if label_path.exists():
                for line in label_path.read_text(encoding="utf-8").splitlines():
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        try:
                            cid = int(parts[0])
                            xc, yc, w, h = (float(v) for v in parts[1:5])
                            if cid in (0, 1) and w > 0 and h > 0:
                                boxes.append(
                                    NormalizedBox(
                                        class_id=cid,
                                        x_center=xc,
                                        y_center=yc,
                                        width=w,
                                        height=h,
                                    )
                                )
                        except ValueError:
                            summary.invalid_boxes += 1

            dest_name = f"human_{img_path.name}"
            dest_img = self.train_images_dir / dest_name
            dest_lbl = self.train_labels_dir / f"{dest_img.stem}.txt"

            shutil.copyfile(img_path, dest_img)
            dest_lbl.write_text(
                "".join(b.to_yolo_line() for b in boxes), encoding="utf-8"
            )
            self.train_hashes.add(info.dhash)

            summary.added_images += 1
            summary.total_boxes += len(boxes)
            counts: dict[str, int] = {"trunk": 0, "bunch": 0}
            for b in boxes:
                counts[CLASS_NAMES[b.class_id]] += 1
                summary.boxes_by_class[CLASS_NAMES[b.class_id]] += 1

            self._log_ingestion(
                dest_name, info.dhash, status="added", box_counts=counts
            )
            summary.details.append(
                {"status": "added", "image": dest_name, "boxes": len(boxes)}
            )

        return summary

    def _log_ingestion(
        self,
        image_name: str,
        dhash: str,
        status: str,
        box_counts: dict[str, int],
    ) -> None:
        record = {
            "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
            "image": image_name,
            "dhash": dhash,
            "status": status,
            "boxes": box_counts,
        }
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Путь к файлу экспорта Label Studio JSON или каталогу меток YOLO",
    )
    parser.add_argument(
        "--images",
        type=Path,
        default=None,
        help="Каталог с исходными кадрами",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/processed/vine"),
        help="Каталог датасета с подкаталогами images/ и labels/",
    )
    args = parser.parse_args(argv)

    importer = AnnotationImporter(args.dataset)
    if args.input.is_file():
        summary = importer.import_label_studio(args.input, args.images)
    elif args.input.is_dir() and args.images:
        summary = importer.import_yolo_dir(args.images, args.input)
    else:
        raise ValueError(
            f"Неподдерживаемый формат входа: {args.input}. Укажите файл .json или пару каталогов."
        )

    print(
        json.dumps(
            {
                "total_tasks": summary.total_tasks,
                "added_images": summary.added_images,
                "skipped_duplicates": summary.skipped_duplicates,
                "rejected_leaks": summary.rejected_leaks,
                "total_boxes": summary.total_boxes,
                "boxes_by_class": summary.boxes_by_class,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
