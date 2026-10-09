"""Подготовка винограда из PlantVillage (Hughes, Salathé, 2015).

Запуск из каталога ml:  uv run python -m vitiguard_ml.data.plantvillage
"""

import argparse
import subprocess
from pathlib import Path

from vitiguard_ml.data.folders import import_folders, print_summary
from vitiguard_ml.data.manifest import write_manifest

REPOSITORY = "https://github.com/spMohanty/PlantVillage-Dataset.git"
SOURCE = "plantvillage"
# Папки датасета → коды классов проекта. Милдью и оидиума в PlantVillage нет
CLASS_MAP = {
    "Grape___Black_rot": "black_rot",
    "Grape___Esca_(Black_Measles)": "esca",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": "leaf_blight",
    "Grape___healthy": "healthy",
}


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=Path("data/raw/plantvillage"))
    parser.add_argument("--out", type=Path, default=Path("data/processed/plantvillage"))
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()

    color_root = (
        args.raw / "raw" / "color" if args.skip_download else download(args.raw)
    )
    rows, skipped = import_folders(color_root, CLASS_MAP, SOURCE, args.out)
    write_manifest(rows, args.out / "manifest.csv")
    print_summary(rows, skipped)


if __name__ == "__main__":
    main()
