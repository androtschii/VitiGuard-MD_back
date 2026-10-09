import shutil
from collections import Counter
from pathlib import Path

from vitiguard_ml.data.images import IMAGE_SUFFIXES, read_image_info
from vitiguard_ml.data.manifest import ManifestRow, file_sha256


def import_folders(
    raw_root: Path, class_map: dict[str, str], source: str, out_root: Path
) -> tuple[list[ManifestRow], Counter[str]]:
    """Импорт набора вида «папка = класс». Папки, которых нет в class_map, не
    берутся. Изображения копируются в out_root/<класс>/<sha256[:16]>.<расширение>.

    Имя по хэшу содержимого: одинаковые файлы совпадают по имени и в набор
    попадают один раз — иначе копия могла бы оказаться и в обучающей, и в тестовой
    выборке, завышая оценку качества."""
    rows: list[ManifestRow] = []
    skipped: Counter[str] = Counter()
    seen: set[str] = set()
    for folder, label in class_map.items():
        class_dir = raw_root / folder
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Нет папки класса {class_dir}")
        for file in sorted(class_dir.iterdir()):
            if file.suffix.lower() not in IMAGE_SUFFIXES:
                skipped["не изображение"] += 1
                continue
            info = read_image_info(file)
            if info is None:
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
                    source=source,
                    sha256=digest,
                    dhash=info.dhash,
                    width=info.width,
                    height=info.height,
                )
            )
    return rows, skipped


def print_summary(rows: list[ManifestRow], skipped: Counter[str]) -> None:
    counts = Counter(row.label for row in rows)
    for label, count in sorted(counts.items()):
        print(f"{label:16} {count:6}")
    print(f"{'итого':16} {len(rows):6}")
    for reason, count in skipped.items():
        print(f"пропущено ({reason}): {count}")
