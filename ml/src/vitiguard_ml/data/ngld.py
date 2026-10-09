"""Подготовка Niphad Grape Leaf Disease Dataset (NGLD), Mendeley Data, CC BY 4.0.

Столовый виноград, снимки на телефон в реальных условиях; в отличие от
PlantVillage, есть милдью и оидиум. DOI 10.17632/8nnd2ypcv3.5

Запуск из каталога ml:  uv run python -m vitiguard_ml.data.ngld
"""

import argparse
import urllib.request
import zipfile
from pathlib import Path

from vitiguard_ml.data.folders import import_folders, print_summary
from vitiguard_ml.data.manifest import write_manifest

DATASET_ID = "8nnd2ypcv3"
VERSION = 5
DOWNLOAD_URL = (
    f"https://data.mendeley.com/public-api/zip/{DATASET_ID}/download/{VERSION}"
)
SOURCE = "ngld"
INNER_ROOT = Path(
    "Niphad Grape Leaf Disease Dataset (NGLD)",
    "Niphad Grape Leaf Disease Dataset (NGLD)",
    "Grapes Disease Dataset",
)
# «Bacterial Leaf Spot» (100 фото) не берётся: такого класса в системе нет, и
# для виноградников Молдовы эта болезнь не основная
CLASS_MAP = {
    "Downy Mildew": "downy_mildew",
    "Powdery Mildew": "powdery_mildew",
    "Healthy Leaves": "healthy",
}


def download(raw_dir: Path) -> Path:
    """Скачивает и распаковывает архив; повторный запуск использует скачанное."""
    archive = raw_dir / f"ngld-{VERSION}.zip"
    if not archive.exists():
        raw_dir.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(DOWNLOAD_URL, archive)
    extracted = raw_dir / f"ngld-{VERSION}"
    if not extracted.exists():
        with zipfile.ZipFile(archive) as zip_file:
            zip_file.extractall(extracted)
    return extracted / INNER_ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=Path("data/raw"))
    parser.add_argument("--out", type=Path, default=Path("data/processed/ngld"))
    args = parser.parse_args()

    rows, skipped = import_folders(download(args.raw), CLASS_MAP, SOURCE, args.out)
    write_manifest(rows, args.out / "manifest.csv")
    print_summary(rows, skipped)


if __name__ == "__main__":
    main()
