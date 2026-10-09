import io
import zipfile
from pathlib import Path

from PIL import Image

from vitiguard_ml.data.hermos import (
    MIN_SIDE,
    Annotation,
    crop_window,
    parse_annotations,
    prepare,
    select_crops,
)

SIZE = (6000, 4000)


def test_window_is_square_with_context() -> None:
    left, top, right, bottom = crop_window((1000, 1000, 1200, 1100), SIZE)

    assert right - left == bottom - top == 600  # 3 × длинная сторона рамки
    assert (left + right) // 2 == 1100
    assert (top + bottom) // 2 == 1050


def test_small_box_gets_minimal_window() -> None:
    left, _, right, _ = crop_window((1000, 1000, 1040, 1010), SIZE)

    assert right - left == MIN_SIDE


def test_window_near_edge_is_shifted_inside() -> None:
    left, top, right, bottom = crop_window((5950, 3980, 6000, 4000), SIZE)

    assert right == 6000
    assert bottom == 4000
    assert (right - left, bottom - top) == (MIN_SIDE, MIN_SIDE)


def test_window_never_exceeds_image() -> None:
    left, top, right, bottom = crop_window((0, 0, 3000, 3000), (2000, 1500))

    assert right - left == bottom - top == 1500
    assert left >= 0
    assert top >= 0
    assert right <= 2000
    assert bottom <= 1500


def test_crop_with_another_class_nearby_is_dropped() -> None:
    annotations = [
        Annotation("healthy", (1000, 1000, 1200, 1200)),
        Annotation("downy mildew", (1250, 1050, 1300, 1100)),  # внутри окна здорового
        Annotation("powdery mildew", (4000, 3000, 4200, 3200)),
    ]

    crops = select_crops(annotations, SIZE)

    labels = [label for label, _ in crops]
    assert "healthy" not in labels
    assert "powdery_mildew" in labels


def test_overlapping_crops_of_one_class_are_kept_once() -> None:
    annotations = [
        Annotation("powdery mildew", (1000, 1000, 1200, 1200)),
        Annotation("powdery mildew", (1150, 1100, 1350, 1300)),
        Annotation("powdery mildew", (3000, 3000, 3200, 3200)),
    ]

    assert len(select_crops(annotations, SIZE)) == 2


def test_unknown_classes_are_ignored() -> None:
    annotations = [
        Annotation("dead arm", (100, 100, 300, 300)),
        Annotation("dog", (2000, 2000, 2100, 2100)),
    ]

    assert select_crops(annotations, SIZE) == []


def voc(
    width: int, height: int, objects: list[tuple[str, tuple[int, int, int, int]]]
) -> bytes:
    parts = [
        f"<object><name>{name}</name><bndbox><xmin>{x1}</xmin><ymin>{y1}</ymin>"
        f"<xmax>{x2}</xmax><ymax>{y2}</ymax></bndbox></object>"
        for name, (x1, y1, x2, y2) in objects
    ]
    return (
        f"<annotation><size><width>{width}</width><height>{height}</height></size>"
        f"{''.join(parts)}</annotation>"
    ).encode()


def test_parse_pascal_voc() -> None:
    size, annotations = parse_annotations(
        voc(800, 600, [("healthy", (10, 20, 110, 120))])
    )

    assert size == (800, 600)
    assert annotations == [Annotation("healthy", (10, 20, 110, 120))]


def test_prepare_writes_field_crops(tmp_path: Path) -> None:
    image = Image.new("RGB", (1600, 1200), (30, 120, 30))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    archive = tmp_path / "hermos.zip"
    with zipfile.ZipFile(archive, "w") as zip_file:
        zip_file.writestr("Images/leaf1.jpg", buffer.getvalue())
        zip_file.writestr(
            "Images/leaf1.xml",
            voc(
                1600,
                1200,
                [
                    ("healthy", (100, 100, 250, 250)),
                    ("downy mildew", (1200, 800, 1260, 830)),
                ],
            ),
        )

    rows, skipped = prepare(archive, tmp_path / "out")

    assert sorted(row.label for row in rows) == ["downy_mildew", "healthy"]
    assert not skipped
    for row in rows:
        assert row.source == "hermos"
        assert (row.width, row.height) == (256, 256)
        with Image.open(tmp_path / "out" / row.path) as crop:
            assert crop.size == (256, 256)
