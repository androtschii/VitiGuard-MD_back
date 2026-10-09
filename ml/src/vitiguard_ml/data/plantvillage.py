"""Подготовка винограда из PlantVillage (Hughes, Salathé, 2015).

Запуск из каталога ml:  uv run python -m vitiguard_ml.data.plantvillage
"""

import argparse
import shutil
import subprocess
from collections import Counter
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from vitiguard_ml.data.manifest import ManifestRow, file_sha256, write_manifest

REPOSITORY = "https://github.com/spMohanty/PlantVillage-Dataset.git"
SOURCE = "plantvillage"
# Папки датасета → коды классов проекта. Милдью и оидиума в PlantVillage нет
CLASS_MAP = {
    "Grape___Black_rot": "black_rot",
    "Grape___Esca_(Black_Measles)": "esca",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": "leaf_blight",
    "Grape___healthy": "healthy",
}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def download(dest: Path) -> Path:
    """Скачивает только четыре папки винограда (цветные фото), а не весь датасет:
    частичный клон без истории и без чужих файлов."""
    if not dest.exists():
        subprocess.run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "--filter=blob:none",
                "--sparse",
                REPOSITORY,
                str(dest),
            ],
            check=True,
        )
    folders = [f"raw/color/{name}" for name in CLASS_MAP]
    subprocess.run(
        ["git", "-C", str(dest), "sparse-checkout", "set", *folders], check=True
    )
    return dest / "raw" / "color"


def _read_size(path: Path) -> tuple[int, int] | None:
    try:
        with Image.open(path) as image:
            image.load()
            return image.size
    except (UnidentifiedImageError, OSError):
        return None


def prepare(raw_root: Path, out_root: Path) -> tuple[list[ManifestRow], Counter[str]]:
    """Копирует изображения в out_root/<класс>/<sha256[:16]>.<расширение> и
    возвращает строки манифеста и счётчик пропущенных файлов по причинам.

    Имя файла — по хэшу содержимого: одинаковые изображения совпадают по имени,
    и дубликат не попадёт в набор дважды (иначе он мог бы оказаться и в обучающей,
    и в тестовой выборке, завышая оценку качества)."""
    rows: list[ManifestRow] = []
    skipped: Counter[str] = Counter()
    seen: set[str] = set()
    for folder, label in CLASS_MAP.items():
        class_dir = raw_root / folder
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Нет папки класса {class_dir}")
        for file in sorted(class_dir.iterdir()):
            if file.suffix.lower() not in IMAGE_SUFFIXES:
                skipped["не изображение"] += 1
                continue
            size = _read_size(file)
            if size is None:
                skipped["повреждён"] += 1
                continue
            digest = file_sha256(file)
            if digest in seen:
                skipped["дубликат"] += 1
                continue
            seen.add(digest)
            relative = Path(label) / f"{digest[:16]}{file.suffix.lower()}"
            target = out_root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(file, target)
            rows.append(
                ManifestRow(
                    path=relative.as_posix(),
                    label=label,
                    source=SOURCE,
                    sha256=digest,
                    width=size[0],
                    height=size[1],
                )
            )
    return rows, skipped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=Path("data/raw/plantvillage"))
    parser.add_argument("--out", type=Path, default=Path("data/processed/plantvillage"))
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()

    color_root = (
        args.raw / "raw" / "color" if args.skip_download else download(args.raw)
    )
    rows, skipped = prepare(color_root, args.out)
    write_manifest(rows, args.out / "manifest.csv")

    counts = Counter(row.label for row in rows)
    for label, count in sorted(counts.items()):
        print(f"{label:12} {count:6}")
    print(f"{'итого':12} {len(rows):6}")
    for reason, count in skipped.items():
        print(f"пропущено ({reason}): {count}")


if __name__ == "__main__":
    main()
