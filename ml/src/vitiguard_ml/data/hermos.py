"""Полевой тестовый набор из HERMOS (Mendeley Data, CC BY 4.0).

Снимки сделаны прямо в виноградниках (Западная Анатолия, Турция): в кадре много
листьев, а симптомы обведены рамками (Pascal VOC). Каждая рамка вырезается
вместе с окружением, и получается снимок листа в полевых условиях — фон, свет,
соседние листья. Набор используется только для теста: так видно, насколько
модель, обученная на лабораторных снимках, ошибается в поле.
DOI 10.17632/j4xs3kh3fd.2

Запуск из каталога ml:  uv run python -m vitiguard_ml.data.hermos
"""

import argparse
import hashlib
import io
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from vitiguard_ml.data.folders import print_summary
from vitiguard_ml.data.images import difference_hash
from vitiguard_ml.data.manifest import ManifestRow, write_manifest

DATASET_ID = "j4xs3kh3fd"
VERSION = 2
DOWNLOAD_URL = (
    f"https://data.mendeley.com/public-api/zip/{DATASET_ID}/download/{VERSION}"
)
SOURCE = "hermos"
# «dead arm» (фомопсис) и случайные «dog» не берутся: таких классов в системе нет
CLASS_MAP = {
    "healthy": "healthy",
    "powdery mildew": "powdery_mildew",
    "downy mildew": "downy_mildew",
}
# Окно вырезки: в CONTEXT раз больше рамки, но не меньше MIN_SIDE пикселей —
# в кадр попадает лист с окружением, а не только пятно
CONTEXT = 3
MIN_SIDE = 384
OUTPUT_SIZE = 256

Box = tuple[int, int, int, int]


@dataclass(frozen=True)
class Annotation:
    name: str
    box: Box


def crop_window(box: Box, image_size: tuple[int, int]) -> Box:
    """Квадратное окно вокруг центра рамки. У края снимка окно сдвигается внутрь,
    а не обрезается: иначе на вырезке появились бы пустые полосы."""
    width, height = image_size
    x1, y1, x2, y2 = box
    side = min(max(MIN_SIDE, CONTEXT * max(x2 - x1, y2 - y1)), width, height)
    left = min(max((x1 + x2) // 2 - side // 2, 0), width - side)
    top = min(max((y1 + y2) // 2 - side // 2, 0), height - side)
    return left, top, left + side, top + side


def _contains_point(window: Box, x: float, y: float) -> bool:
    left, top, right, bottom = window
    return left <= x < right and top <= y < bottom


def _center(box: Box) -> tuple[float, float]:
    return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2


def select_crops(
    annotations: list[Annotation], image_size: tuple[int, int]
) -> list[tuple[str, Box]]:
    """Выбирает вырезки снимка: (класс проекта, окно).

    Вырезка отбрасывается, если в окно попадает центр рамки другого класса: у
    «здорового» листа рядом с пятном милдью метка была бы неверной. Вырезка
    одного класса, чей центр уже внутри принятого окна, тоже не берётся —
    иначе в наборе было бы много почти одинаковых кадров."""
    accepted: list[tuple[str, Box]] = []
    for annotation in annotations:
        label = CLASS_MAP.get(annotation.name)
        if label is None:
            continue
        window = crop_window(annotation.box, image_size)
        conflict = any(
            other.name != annotation.name
            and _contains_point(window, *_center(other.box))
            for other in annotations
        )
        overlaps = any(
            accepted_label == label
            and _contains_point(accepted_window, *_center(annotation.box))
            for accepted_label, accepted_window in accepted
        )
        if not conflict and not overlaps:
            accepted.append((label, window))
    return accepted


def parse_annotations(xml_bytes: bytes) -> tuple[tuple[int, int], list[Annotation]]:
    root = ET.fromstring(xml_bytes)
    size = root.find("size")
    if size is None:
        raise ValueError("В разметке нет размера снимка")
    image_size = (int(size.findtext("width", "0")), int(size.findtext("height", "0")))
    annotations = []
    for obj in root.iter("object"):
        bndbox = obj.find("bndbox")
        if bndbox is None:
            continue
        x1, y1, x2, y2 = (
            int(float(bndbox.findtext(key, "0")))
            for key in ("xmin", "ymin", "xmax", "ymax")
        )
        annotations.append(
            Annotation(obj.findtext("name", "").strip(), (x1, y1, x2, y2))
        )
    return image_size, annotations


def download(raw_dir: Path) -> Path:
    archive = raw_dir / f"hermos-{VERSION}.zip"
    if not archive.exists():
        raw_dir.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(DOWNLOAD_URL, archive)
    return archive


def prepare(archive: Path, out_root: Path) -> tuple[list[ManifestRow], Counter[str]]:
    rows: list[ManifestRow] = []
    skipped: Counter[str] = Counter()
    with zipfile.ZipFile(archive) as zip_file:
        names = zip_file.namelist()
        images = {
            name.rsplit(".", 1)[0]: name
            for name in names
            if name.lower().endswith((".jpg", ".jpeg", ".png"))
        }
        for xml_name in sorted(name for name in names if name.endswith(".xml")):
            image_name = images.get(xml_name.rsplit(".", 1)[0])
            if image_name is None:
                skipped["нет снимка"] += 1
                continue
            image_size, annotations = parse_annotations(zip_file.read(xml_name))
            crops = select_crops(annotations, image_size)
            dropped = sum(a.name in CLASS_MAP for a in annotations) - len(crops)
            if dropped:
                skipped["рамка отброшена"] += dropped
            if not crops:
                continue
            with Image.open(io.BytesIO(zip_file.read(image_name))) as source:
                image = source.convert("RGB")
            if image.size != image_size:
                skipped["размер не совпадает с разметкой"] += 1
                continue
            for label, window in crops:
                crop = image.crop(window).resize(
                    (OUTPUT_SIZE, OUTPUT_SIZE), Image.Resampling.LANCZOS
                )
                buffer = io.BytesIO()
                crop.save(buffer, format="JPEG", quality=95)
                data = buffer.getvalue()
                digest = hashlib.sha256(data).hexdigest()
                relative = Path(label) / f"{digest[:16]}.jpg"
                (out_root / relative).parent.mkdir(parents=True, exist_ok=True)
                (out_root / relative).write_bytes(data)
                rows.append(
                    ManifestRow(
                        path=relative.as_posix(),
                        label=label,
                        source=SOURCE,
                        sha256=digest,
                        dhash=difference_hash(crop),
                        width=OUTPUT_SIZE,
                        height=OUTPUT_SIZE,
                    )
                )
    return rows, skipped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=Path("data/raw"))
    parser.add_argument("--out", type=Path, default=Path("data/processed/hermos"))
    args = parser.parse_args()

    rows, skipped = prepare(download(args.raw), args.out)
    write_manifest(rows, args.out / "manifest.csv")
    print_summary(rows, skipped)


if __name__ == "__main__":
    main()
