"""Набор «ствол + гроздь» для модели лозы (ml/pr-023).

Источники с открытыми лицензиями, рамки объектов:
- INESC TEC «Grape bunch and vine trunk dataset» (Aguiar, Magalhães, Zenodo,
  DOI 10.5281/zenodo.5114141, CC BY 4.0): стволы и грозди, Pascal VOC;
- Embrapa WGISD (CC BY-NC 4.0, ml/pr-022): грозди, YOLO.

В архиве INESC рядом с кадрами лежат их искусственные копии (поворот,
отражение, шум): они отбрасываются, иначе копия кадра попала бы и в обучение,
и в проверку. Тепловизионные кадры тоже не берутся — это другая камера.
Кадры INESC идут подряд из видео, поэтому выборки делятся блоками соседних
кадров, а не случайно.

Запуск из каталога ml:  uv run python -m vitiguard_ml.data.vine
"""

import argparse
import json
import re
import shutil
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

from vitiguard_ml.data.wgisd import download as download_wgisd
from vitiguard_ml.data.wgisd import read_list, split_names

CLASSES = ("trunk", "bunch")
# Имена классов в разметке INESC: две величины гроздей — одна «гроздь»,
# опечатка «truk» — ствол
INESC_CLASSES = {
    "trunk": 0,
    "truk": 0,
    "medium_grape_bunch": 1,
    "tiny_grape_bunch": 1,
}
AUGMENTED = re.compile(
    r"^(multiply|multiplyrandom|flip|angle-?\d+|scale\d+|translaterandom|"
    r"noiserandom|huesatur|randomcombination\d+|rotated)_"
)
BLOCK = 50
INESC_ARCHIVE = "inesc/grape_trunk_dataset.zip"


def is_original(name: str) -> bool:
    return not AUGMENTED.match(name) and not name.startswith("thermal")


def voc_to_yolo(xml: bytes) -> list[str]:
    """Рамки Pascal VOC → строки YOLO «класс cx cy w h» в долях кадра."""
    root = ET.fromstring(xml)
    size = root.find("size")
    assert size is not None
    width = float(size.findtext("width", "0"))
    height = float(size.findtext("height", "0"))
    lines = []
    for item in root.findall("object"):
        label = INESC_CLASSES.get(item.findtext("name", ""))
        box = item.find("bndbox")
        if label is None or box is None:
            continue
        x1, y1, x2, y2 = (
            float(box.findtext(key, "0")) for key in ("xmin", "ymin", "xmax", "ymax")
        )
        x1, x2 = max(0.0, min(x1, x2)), min(width, max(x1, x2))
        y1, y2 = max(0.0, min(y1, y2)), min(height, max(y1, y2))
        if x2 - x1 < 1 or y2 - y1 < 1:
            continue
        lines.append(
            f"{label} {(x1 + x2) / 2 / width:.6f} {(y1 + y2) / 2 / height:.6f} "
            f"{(x2 - x1) / width:.6f} {(y2 - y1) / height:.6f}"
        )
    return lines


def block_split(names: list[str]) -> dict[str, str]:
    """Кадр → выборка по его блоку из BLOCK соседних кадров своей съёмки:
    каждый десятый блок — тест, следующий за ним — проверка, остальные —
    обучение. Без случайности: повторный запуск даёт то же разбиение."""
    by_group: dict[str, list[str]] = {}
    for name in sorted(names):
        by_group.setdefault(re.split(r"[\d_]", name)[0], []).append(name)
    result = {}
    for group in by_group.values():
        for index, name in enumerate(group):
            block = index // BLOCK
            result[name] = (
                "test" if block % 10 == 0 else "val" if block % 10 == 1 else "train"
            )
    return result


def prepare_inesc(archive: Path, out: Path) -> dict[str, int]:
    counts = {"train": 0, "val": 0, "test": 0}
    with zipfile.ZipFile(archive) as zip_file:
        xmls = {
            Path(name).stem: name
            for name in zip_file.namelist()
            if name.endswith(".xml") and is_original(Path(name).stem)
        }
        images = {
            Path(name).stem: name
            for name in zip_file.namelist()
            if name.lower().endswith(".jpg")
        }
        splits = block_split([name for name in xmls if name in images])
        for name, split in splits.items():
            (out / "images" / split).mkdir(parents=True, exist_ok=True)
            (out / "labels" / split).mkdir(parents=True, exist_ok=True)
            with (
                zip_file.open(images[name]) as source,
                (out / "images" / split / f"inesc_{name}.jpg").open("wb") as target,
            ):
                shutil.copyfileobj(source, target)
            (out / "labels" / split / f"inesc_{name}.txt").write_text(
                "\n".join(voc_to_yolo(zip_file.read(xmls[name])))
            )
            counts[split] += 1
    return counts


def prepare_wgisd(root: Path, out: Path) -> dict[str, int]:
    """Рамки WGISD: класс 0 «гроздь» становится классом 1 общего набора."""
    splits = split_names(read_list(root / "train.txt"), read_list(root / "test.txt"))
    counts = {}
    for split, names in splits.items():
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        for name in names:
            shutil.copyfile(
                root / "data" / f"{name}.jpg",
                out / "images" / split / f"wgisd_{name}.jpg",
            )
            lines = (root / "data" / f"{name}.txt").read_text().split("\n")
            (out / "labels" / split / f"wgisd_{name}.txt").write_text(
                "\n".join("1" + line[1:] for line in lines if line.startswith("0 "))
            )
        counts[split] = len(names)
    return counts


def write_config(out: Path, sources: dict[str, dict[str, int]]) -> None:
    names = "".join(f"  {index}: {name}\n" for index, name in enumerate(CLASSES))
    (out / "data.yaml").write_text(
        f"path: {out.resolve().as_posix()}\n"
        f"train: images/train\nval: images/val\ntest: images/test\nnames:\n{names}"
    )
    (out / "sources.json").write_text(
        json.dumps(sources, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=Path("data/raw"))
    parser.add_argument("--out", type=Path, default=Path("data/processed/vine"))
    args = parser.parse_args()

    sources = {"inesc": prepare_inesc(args.raw / INESC_ARCHIVE, args.out)}
    sources["wgisd"] = prepare_wgisd(download_wgisd(args.raw), args.out)
    write_config(args.out, sources)
    print(json.dumps(sources, ensure_ascii=False))


if __name__ == "__main__":
    main()
