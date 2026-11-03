from pathlib import Path

import numpy as np

from vitiguard_ml.robot.collection import (
    FrameCollector,
    RobotPose,
    is_blurry,
    laplacian_variance,
)


def test_laplacian_variance_sharp_vs_blurry() -> None:
    # Резкое изображение: шахматный узор с резкими границами
    sharp = np.zeros((100, 100, 3), dtype=np.uint8)
    sharp[::2, ::2] = 255
    sharp[1::2, 1::2] = 255

    # Размытое изображение: гладкий градиент
    blurry = np.zeros((100, 100, 3), dtype=np.uint8)
    for i in range(100):
        blurry[i, :] = int(i * 2.5)

    var_sharp = laplacian_variance(sharp)
    var_blurry = laplacian_variance(blurry)

    assert var_sharp > 500.0
    assert var_blurry < 50.0

    is_b, score = is_blurry(blurry, min_variance=80.0)
    assert is_b
    assert score == var_blurry


def test_frame_collector_saves_sharp_and_rejects_blurry(tmp_path: Path) -> None:
    collector = FrameCollector(output_dir=tmp_path / "collector", min_variance=80.0)
    pose = RobotPose(row_id="R01", x=12.5, y=34.2, heading_deg=90.0)

    # 1. Смазанный кадр
    blurry_img = np.zeros((120, 160, 3), dtype=np.uint8)
    res_blurry = collector.process_frame(blurry_img, pose=pose, frame_id="f_blurry")
    assert res_blurry is None
    assert collector.stats["rejected_blurry"] == 1
    assert collector.stats["saved"] == 0

    # 2. Резкий кадр с текстурой
    np.random.seed(42)
    sharp_img = np.random.randint(0, 256, (120, 160, 3), dtype=np.uint8)
    depth = np.full((120, 160), 0.75, dtype=np.float32)

    res_sharp = collector.process_frame(
        sharp_img, pose=pose, depth=depth, frame_id="f_sharp"
    )
    assert res_sharp is not None
    assert collector.stats["saved"] == 1
    assert res_sharp.frame_id == "f_sharp"
    assert (collector.rgb_dir / "f_sharp.jpg").exists()
    assert (collector.depth_dir / "f_sharp.npy").exists()
    assert collector.manifest_file.exists()


def test_frame_collector_rejects_duplicate(tmp_path: Path) -> None:
    collector = FrameCollector(output_dir=tmp_path / "collector", min_variance=10.0)
    pose = RobotPose(row_id="R01", x=12.5, y=34.2, heading_deg=90.0)

    np.random.seed(123)
    img1 = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    # Почти такой же кадр (небольшой шум)
    img2 = np.clip(img1.astype(np.int16) + 2, 0, 255).astype(np.uint8)

    meta1 = collector.process_frame(img1, pose=pose, frame_id="f1")
    assert meta1 is not None

    meta2 = collector.process_frame(img2, pose=pose, frame_id="f2")
    assert meta2 is None
    assert collector.stats["rejected_duplicate"] == 1
    assert collector.stats["saved"] == 1
