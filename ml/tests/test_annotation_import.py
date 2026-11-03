import json
from pathlib import Path

from conftest import make_image

from vitiguard_ml.robot.annotation_import import (
    AnnotationImporter,
    parse_label_studio_result,
)


def test_parse_label_studio_rectangle() -> None:
    item = {
        "type": "rectanglelabels",
        "value": {
            "x": 10.0,
            "y": 20.0,
            "width": 30.0,
            "height": 40.0,
            "rectanglelabels": ["trunk"],
        },
    }
    box = parse_label_studio_result(item)
    assert box is not None
    assert box.class_id == 0
    # Center x: (10 + 15) / 100 = 0.25, center y: (20 + 20) / 100 = 0.40
    assert box.x_center == 0.25
    assert box.y_center == 0.40
    assert box.width == 0.30
    assert box.height == 0.40


def test_parse_label_studio_polygon() -> None:
    item = {
        "type": "polygonlabels",
        "value": {
            "points": [
                [50.0, 50.0],
                [60.0, 50.0],
                [60.0, 70.0],
                [50.0, 70.0],
            ],
            "polygonlabels": ["bunch"],
        },
    }
    box = parse_label_studio_result(item)
    assert box is not None
    assert box.class_id == 1
    # Bounding box: min_x=50, max_x=60, min_y=50, max_y=70 -> cx=0.55, cy=0.60, w=0.10, h=0.20
    assert box.x_center == 0.55
    assert box.y_center == 0.60
    assert box.width == 0.10
    assert box.height == 0.20


def test_parse_unknown_label_returns_none() -> None:
    item = {
        "type": "rectanglelabels",
        "value": {
            "x": 10.0,
            "y": 20.0,
            "width": 10.0,
            "height": 10.0,
            "rectanglelabels": ["unknown_class"],
        },
    }
    assert parse_label_studio_result(item) is None


def test_import_label_studio_flow(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    images_dir = tmp_path / "raw_frames"
    images_dir.mkdir(parents=True)

    img_path = images_dir / "frame_01.jpg"
    make_image(img_path, seed=1)

    export_json = tmp_path / "export.json"
    task = {
        "id": 42,
        "data": {"image": "frame_01.jpg"},
        "annotations": [
            {
                "result": [
                    {
                        "type": "rectanglelabels",
                        "value": {
                            "x": 20.0,
                            "y": 10.0,
                            "width": 15.0,
                            "height": 50.0,
                            "rectanglelabels": ["trunk"],
                        },
                    },
                    {
                        "type": "rectanglelabels",
                        "value": {
                            "x": 40.0,
                            "y": 60.0,
                            "width": 10.0,
                            "height": 20.0,
                            "rectanglelabels": ["bunch"],
                        },
                    },
                ]
            }
        ],
    }
    export_json.write_text(json.dumps([task]), encoding="utf-8")

    importer = AnnotationImporter(dataset_dir)
    summary = importer.import_label_studio(export_json, images_dir)

    assert summary.total_tasks == 1
    assert summary.added_images == 1
    assert summary.total_boxes == 2
    assert summary.boxes_by_class["trunk"] == 1
    assert summary.boxes_by_class["bunch"] == 1

    dest_img = dataset_dir / "images" / "train" / "human_frame_01.jpg"
    dest_lbl = dataset_dir / "labels" / "train" / "human_frame_01.txt"
    assert dest_img.exists()
    assert dest_lbl.exists()

    label_content = dest_lbl.read_text(encoding="utf-8").strip().splitlines()
    assert len(label_content) == 2
    assert label_content[0].startswith("0 ")
    assert label_content[1].startswith("1 ")

    # Проверка журнала пополнения
    log_file = dataset_dir / "ingestion_log.jsonl"
    assert log_file.exists()
    log_lines = log_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(log_lines) == 1
    record = json.loads(log_lines[0])
    assert record["status"] == "added"
    assert record["image"] == "human_frame_01.jpg"


def test_anti_leakage_rejects_test_duplicates(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    test_dir = dataset_dir / "images" / "test"
    test_dir.mkdir(parents=True)

    # Кадр в тестовой выборке
    test_img = test_dir / "benchmark_01.jpg"
    make_image(test_img, seed=99)

    images_dir = tmp_path / "incoming"
    images_dir.mkdir(parents=True)

    # Кандидат для обучения — идентичен кадру из теста
    leak_img = images_dir / "leak_frame.jpg"
    make_image(leak_img, seed=99)

    export_json = tmp_path / "export_leak.json"
    task = {
        "id": 100,
        "data": {"image": "leak_frame.jpg"},
        "annotations": [
            {
                "result": [
                    {
                        "type": "rectanglelabels",
                        "value": {
                            "x": 10.0,
                            "y": 10.0,
                            "width": 20.0,
                            "height": 30.0,
                            "rectanglelabels": ["trunk"],
                        },
                    }
                ]
            }
        ],
    }
    export_json.write_text(json.dumps([task]), encoding="utf-8")

    importer = AnnotationImporter(dataset_dir)
    summary = importer.import_label_studio(export_json, images_dir)

    assert summary.rejected_leaks == 1
    assert summary.added_images == 0
    # Кадр не должен попасть в обучающую выборку!
    assert not (dataset_dir / "images" / "train" / "human_leak_frame.jpg").exists()

    # В журнале зафиксировано отклонение
    log_file = dataset_dir / "ingestion_log.jsonl"
    assert log_file.exists()
    record = json.loads(log_file.read_text(encoding="utf-8").strip())
    assert record["status"] == "rejected_test_leak"


def test_duplicate_skipping_in_train(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    images_dir = tmp_path / "incoming"
    images_dir.mkdir(parents=True)

    img_path = images_dir / "frame_dup.jpg"
    make_image(img_path, seed=55)

    export_json = tmp_path / "export_dup.json"
    task = {
        "id": 1,
        "data": {"image": "frame_dup.jpg"},
        "annotations": [],
    }
    export_json.write_text(json.dumps([task]), encoding="utf-8")

    importer = AnnotationImporter(dataset_dir)
    sum1 = importer.import_label_studio(export_json, images_dir)
    assert sum1.added_images == 1
    assert sum1.skipped_duplicates == 0

    # Повторный импорт того же кадра
    sum2 = importer.import_label_studio(export_json, images_dir)
    assert sum2.added_images == 0
    assert sum2.skipped_duplicates == 1


def test_import_yolo_dir(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    src_images = tmp_path / "yolo_in" / "images"
    src_labels = tmp_path / "yolo_in" / "labels"
    src_images.mkdir(parents=True)
    src_labels.mkdir(parents=True)

    img1 = src_images / "vine_001.jpg"
    make_image(img1, seed=12)
    (src_labels / "vine_001.txt").write_text(
        "0 0.5 0.5 0.2 0.8\n1 0.7 0.3 0.1 0.2\n", encoding="utf-8"
    )

    importer = AnnotationImporter(dataset_dir)
    summary = importer.import_yolo_dir(src_images, src_labels)

    assert summary.total_tasks == 1
    assert summary.added_images == 1
    assert summary.total_boxes == 2
    assert (dataset_dir / "images" / "train" / "human_vine_001.jpg").exists()
    assert (dataset_dir / "labels" / "train" / "human_vine_001.txt").exists()
