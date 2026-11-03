from pathlib import Path

from conftest import make_image
from PIL import Image

from vitiguard_ml.data.images import hamming, read_image_info


def test_dhash_survives_resize_and_recompression(tmp_path: Path) -> None:
    original = tmp_path / "original.png"
    make_image(original, seed=5, size=(256, 256))
    copy = tmp_path / "copy.jpg"
    Image.open(original).resize((128, 128)).save(copy, quality=70)

    first, second = read_image_info(original), read_image_info(copy)

    assert first is not None
    assert second is not None
    assert hamming(first.dhash, second.dhash) <= 2


def test_different_pictures_have_different_dhash(tmp_path: Path) -> None:
    make_image(tmp_path / "a.png", seed=1, size=(256, 256))
    make_image(tmp_path / "b.png", seed=9, size=(256, 256))

    first, second = read_image_info(tmp_path / "a.png"), read_image_info(
        tmp_path / "b.png"
    )

    assert first is not None
    assert second is not None
    assert hamming(first.dhash, second.dhash) > 10


def test_unreadable_file_gives_none(tmp_path: Path) -> None:
    broken = tmp_path / "broken.jpg"
    broken.write_bytes(b"\xff\xd8 not really a jpeg")

    assert read_image_info(broken) is None


def test_hamming_counts_different_bits() -> None:
    assert hamming("0000", "0000") == 0
    assert hamming("0001", "0000") == 1
    assert hamming("ffff", "0000") == 16
