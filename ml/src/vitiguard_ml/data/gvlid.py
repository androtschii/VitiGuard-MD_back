"""Подготовка набора Grapevine Leaf Disease Dataset (GVLiD), Mendeley Data, CC BY 4.0.

Полевой набор снимков виноградников штата Махараштра, Индия. 3 477 фото 1080×1080,
4 класса: чёрная гниль, эска, здоровый лист, пятнистость.
DOI 10.17632/wkymf8bhcg.5

Важно: названия папок набора верны, а в приложенном metadata.json поля Disease_Type
и Plant_Health_Status у части снимков перепутаны (например, 671 снимок в папке
'leaf blight' ошибочно помечен как 'Healthy'). Поэтому метка класса берётся
исключительно из каталога (CLASS_MAP), а из metadata.json извлекается только
виноградник Location.Vineyard (28 виноградников) для исключения утечки кадров
одного куста между обучающей и тестовой выборками (vineyards.csv).

Запуск из каталога ml:  uv run python -m vitiguard_ml.data.gvlid
"""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from vitiguard_ml.data.folders import import_folders, print_summary
from vitiguard_ml.data.manifest import ManifestRow, write_manifest

SOURCE = "gvlid"
CLASS_MAP = {
    "Black rot": "black_rot",
    "esca": "esca",
    "healthy": "healthy",
    "leaf blight": "leaf_blight",
}
IMAGE_SIZE = 256
JPEG_QUALITY = 92


def _load_metadata(raw_root: Path) -> dict[str, str]:
    """Загружает привязку файла к винограднику из metadata.json."""
    meta_path = raw_root / "metadata.json"
    if not meta_path.exists() and (raw_root.parent / "metadata.json").exists():
        meta_path = raw_root.parent / "metadata.json"

    if not meta_path.exists():
        print(f"Предупреждение: {meta_path} не найден, виноградники будут 'unknown'")
        return {}

    items = json.loads(meta_path.read_text(encoding="utf-8"))
    vineyard_by_name: dict[str, str] = {}
    if isinstance(items, list):
        for item in items:
            if not isinstance(item, dict):
                continue
            filename = item.get("image_filename")
            loc = item.get("Location")
            vineyard = loc.get("Vineyard") if isinstance(loc, dict) else None
            val = str(vineyard) if vineyard else "unknown"
            if filename:
                vineyard_by_name[filename] = val
                vineyard_by_name[filename.lower()] = val
    return vineyard_by_name


def prepare(raw_root: Path, out_root: Path) -> tuple[list[ManifestRow], Counter[str]]:
    """Импортирует снимки GVLiD с уменьшением до 256×256 и формирует vineyards.csv."""
    vineyard_lookup = _load_metadata(raw_root)
    raw_names: dict[str, str] = {}
    rows, skipped = import_folders(
        raw_root,
        CLASS_MAP,
        SOURCE,
        out_root,
        resize=IMAGE_SIZE,
        quality=JPEG_QUALITY,
        raw_names=raw_names,
    )

    vineyards: list[tuple[str, str]] = []
    for row in rows:
        orig_name = raw_names.get(row.path)
        v = (
            vineyard_lookup.get(orig_name)
            or (vineyard_lookup.get(orig_name.lower()) if orig_name else None)
            if orig_name
            else None
        )
        if v is None:
            print(
                f"Предупреждение: для снимка {orig_name or row.path} нет записи в metadata.json, виноградник unknown"
            )
            v = "unknown"
        vineyards.append((row.path, v))

    out_root.mkdir(parents=True, exist_ok=True)
    vineyards_csv = out_root / "vineyards.csv"
    with vineyards_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["path", "vineyard"])
        for path, vineyard in sorted(vineyards, key=lambda entry: entry[0]):
            writer.writerow([path, vineyard])

    write_manifest(rows, out_root / "manifest.csv")
    return rows, skipped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=Path("data/raw/gvlid/DATASET"))
    parser.add_argument("--out", type=Path, default=Path("data/processed/gvlid"))
    args = parser.parse_args()

    rows, skipped = prepare(args.raw, args.out)
    write_manifest(rows, args.out / "manifest.csv")
    print_summary(rows, skipped)


if __name__ == "__main__":
    main()
