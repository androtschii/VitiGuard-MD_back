import json
from pathlib import Path

from conftest import make_image
from PIL import Image

from vitiguard_ml.data.images import difference_hash
from vitiguard_ml.robot.benchmark import (
    BenchmarkBox,
    BenchmarkFrameMetadata,
    check_benchmark_leakage,
    evaluate_sliced_benchmark,
    generate_benchmark_html,
    validate_benchmark,
)


def create_sample_benchmark(
    base_dir: Path,
    num_frames: int = 200,
    num_rows: int = 6,
) -> Path:
    images_dir = base_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    meta_path = base_dir / "metadata.jsonl"

    times = ["morning", "midday", "evening"]
    weathers = ["sunny", "overcast", "shadow"]

    records = []
    for i in range(num_frames):
        fid = f"frame_{i:04d}"
        img_name = f"{fid}.jpg"
        img_p = images_dir / img_name
        make_image(img_p, seed=i)

        with Image.open(img_p) as im:
            dhash = difference_hash(im)

        row_id = f"row_{i % num_rows:02d}"
        time_od = times[i % len(times)]
        w_cond = weathers[i % len(weathers)]

        rec = {
            "frame_id": fid,
            "image_file": img_name,
            "depth_file": None,
            "row_id": row_id,
            "time_of_day": time_od,
            "weather": w_cond,
            "camera_height_mm": 1200,
            "dhash": dhash,
            "boxes": [
                {
                    "class_id": 0,
                    "x_center": 0.3,
                    "y_center": 0.5,
                    "width": 0.15,
                    "height": 0.6,
                    "depth_m": 0.75,
                },
                {
                    "class_id": 1,
                    "x_center": 0.6,
                    "y_center": 0.7,
                    "width": 0.1,
                    "height": 0.2,
                    "depth_m": 1.10,
                },
            ],
        }
        records.append(rec)

    with meta_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    return base_dir


def test_validate_benchmark_valid_set(tmp_path: Path) -> None:
    bench_dir = create_sample_benchmark(tmp_path / "bench", num_frames=200, num_rows=6)
    res = validate_benchmark(bench_dir, expected_frames=200, min_rows=5)

    assert res.is_valid
    assert res.total_frames == 200
    assert res.distinct_rows == 6
    assert res.total_trunks == 200
    assert res.total_bunches == 200
    assert len(res.issues) == 0


def test_validate_benchmark_detects_insufficient_rows(tmp_path: Path) -> None:
    # Только 3 ряда вместо 5
    bench_dir = create_sample_benchmark(
        tmp_path / "bench_bad_rows", num_frames=200, num_rows=3
    )
    res = validate_benchmark(bench_dir, expected_frames=200, min_rows=5)

    assert not res.is_valid
    assert any("разнообразие рядов" in iss for iss in res.issues)


def test_validate_benchmark_detects_missing_image(tmp_path: Path) -> None:
    bench_dir = create_sample_benchmark(
        tmp_path / "bench_miss", num_frames=10, num_rows=5
    )
    # Удаляем одно изображение
    (bench_dir / "images" / "frame_0000.jpg").unlink()

    res = validate_benchmark(bench_dir, expected_frames=10, min_rows=5)
    assert not res.is_valid
    assert any("не найдено" in iss for iss in res.issues)


def test_check_benchmark_leakage(tmp_path: Path) -> None:
    bench_dir = create_sample_benchmark(
        tmp_path / "bench_leak", num_frames=5, num_rows=5
    )
    train_dir = tmp_path / "train_images"
    train_dir.mkdir(parents=True)

    # Кадр 1 в обучающей выборке идентичен кадру эталона
    img_leak = train_dir / "copied_frame.jpg"
    make_image(img_leak, seed=0)

    # Кадр 2 совершенно другой
    img_clean = train_dir / "clean_frame.jpg"
    make_image(img_clean, seed=999)

    leaks = check_benchmark_leakage(bench_dir, train_dir, max_hamming=5)
    assert len(leaks) >= 1
    assert leaks[0]["benchmark_frame"] == "frame_0000"
    assert leaks[0]["train_image"] == "copied_frame.jpg"


def test_evaluate_sliced_benchmark(tmp_path: Path) -> None:
    frame = BenchmarkFrameMetadata(
        frame_id="f1",
        image_file="f1.jpg",
        depth_file=None,
        row_id="row_01",
        time_of_day="morning",
        weather="sunny",
        camera_height_mm=1200,
        dhash="0" * 16,
        boxes=[
            BenchmarkBox(
                class_id=0,
                x_center=0.5,
                y_center=0.5,
                width=0.2,
                height=0.5,
                depth_m=0.7,
            )
        ],
    )
    preds = {
        "f1": [
            {
                "class_id": 0,
                "x_center": 0.51,
                "y_center": 0.49,
                "width": 0.2,
                "height": 0.5,
            }
        ]
    }

    report = evaluate_sliced_benchmark([frame], preds, model_version="v001")
    assert report.model_version == "v001"
    assert report.total_frames == 1
    assert report.overall_map50 == 0.5  # Trunk 1.0, bunch 0.0 (no bunch GT) -> avg 0.5
    assert report.slices_time["morning"] == 1.0
    assert report.slices_weather["sunny"] == 1.0
    assert report.slices_distance["near_under_0.8m"] == 1.0

    html_out = tmp_path / "report.html"
    generate_benchmark_html(report, html_out)
    assert html_out.exists()
    assert "v001" in html_out.read_text(encoding="utf-8")
