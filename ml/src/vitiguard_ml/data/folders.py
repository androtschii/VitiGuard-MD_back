import hashlib
import io
import shutil
from collections import Counter
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from vitiguard_ml.data.images import IMAGE_SUFFIXES, difference_hash, read_image_info
from vitiguard_ml.data.manifest import ManifestRow, file_sha256


def import_folders(
    raw_root: Path,
    class_map: dict[str, str],
    source: str,
    out_root: Path,
    resize: int | None = None,
    quality: int = 92,
    raw_names: dict[str, str] | None = None,
) -> tuple[list[ManifestRow], Counter[str]]:
    """Импорт набора вида «папка = класс». Папки, которых нет в class_map, не
    берутся. Изображения копируются в out_root/<класс>/<sha256[:16]>.<расширение>.

    Если задан resize, снимки кадрируются по центральному квадрату и билинейно
    уменьшаются до resize×resize с сохранением в JPEG (quality).
    Хэш SHA-256 и dHash вычисляются по сохранённому изображению.

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
            if resize is None:
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
                if raw_names is not None:
                    raw_names[relative.as_posix()] = file.name
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
            else:
                try:
                    with Image.open(file) as img:
                        img.load()
                        rgb = img.convert("RGB")
                except (UnidentifiedImageError, OSError):
                    skipped["повреждён"] += 1
                    continue

                side = min(rgb.width, rgb.height)
                left = (rgb.width - side) // 2
                top = (rgb.height - side) // 2
                cropped = rgb.crop((left, top, left + side, top + side))
                resized = cropped.resize((resize, resize), Image.Resampling.BILINEAR)

                buffer = io.BytesIO()
                resized.save(buffer, format="JPEG", quality=quality)
                data = buffer.getvalue()
                digest = hashlib.sha256(data).hexdigest()
                if digest in seen:
                    skipped["дубликат"] += 1
                    continue
                seen.add(digest)

                relative = Path(label) / f"{digest[:16]}.jpg"
                target = out_root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
                if raw_names is not None:
                    raw_names[relative.as_posix()] = file.name

                rows.append(
                    ManifestRow(
                        path=relative.as_posix(),
                        label=label,
                        source=source,
                        sha256=digest,
                        dhash=difference_hash(resized),
                        width=resize,
                        height=resize,
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
