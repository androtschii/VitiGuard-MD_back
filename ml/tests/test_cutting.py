import numpy as np
import pytest

from vitiguard_ml.robot.cutting import (
    CameraIntrinsics,
    compute_cut_target,
    estimate_stem_direction,
    find_peduncle_base,
)


def test_camera_intrinsics_deproject() -> None:
    cam = CameraIntrinsics(fx=500.0, fy=500.0, cx=320.0, cy=240.0)

    # Точка в оптическом центре на расстоянии 1 метр
    x, y, z = cam.deproject(320.0, 240.0, 1.0)
    assert x == pytest.approx(0.0)
    assert y == pytest.approx(0.0)
    assert z == pytest.approx(1.0)

    # Точка правее и ниже
    x2, y2, z2 = cam.deproject(420.0, 290.0, 2.0)
    assert x2 == pytest.approx((100.0 * 2.0) / 500.0)
    assert y2 == pytest.approx((50.0 * 2.0) / 500.0)
    assert z2 == pytest.approx(2.0)


def test_find_peduncle_base_empty() -> None:
    empty_mask = np.zeros((100, 100), dtype=np.uint8)
    assert find_peduncle_base(empty_mask) is None


def test_find_peduncle_base_finds_top_center() -> None:
    mask = np.zeros((100, 100), dtype=np.uint8)
    # Гроздь: верх на строке 30, столбцы 45..55
    mask[30:35, 45:56] = 1
    # Низ грозди шире
    mask[35:80, 30:70] = 1

    base = find_peduncle_base(mask)
    assert base is not None
    x, y = base
    assert y == 30
    assert 48 <= x <= 52


def test_estimate_stem_direction_vertical() -> None:
    rgb = np.ones((100, 100, 3), dtype=np.uint8) * 200
    # Рисуем вертикальный стебель темно-коричневого цвета (RGB: 60, 40, 20)
    rgb[10:50, 48:52] = [60, 40, 20]

    dx, dy, width = estimate_stem_direction(rgb, base_x=50, base_y=50, search_height=40)

    assert abs(dx) < 0.2
    assert dy < -0.8
    assert width >= 3.0


def test_compute_cut_target_valid() -> None:
    cam = CameraIntrinsics(fx=600.0, fy=600.0, cx=320.0, cy=240.0)

    mask = np.zeros((480, 640), dtype=np.uint8)
    mask[150:350, 300:360] = 1

    depth_map = np.full((480, 640), 0.60, dtype=np.float32)
    rgb = np.full((480, 640, 3), 180, dtype=np.uint8)
    # Стебель
    rgb[100:150, 328:332] = [40, 30, 20]

    target = compute_cut_target(
        mask=mask,
        depth_map=depth_map,
        intrinsics=cam,
        rgb=rgb,
        offset_mm=20.0,
    )

    assert target is not None
    assert target.safe_to_cut
    assert target.pixel[1] < 150  # Срез строго выше основания грозди
    assert 320 <= target.pixel[0] <= 340
    assert target.point_3d[2] == pytest.approx(0.60)
    assert target.confidence > 0.8


def test_compute_cut_target_rejects_out_of_reach() -> None:
    cam = CameraIntrinsics(fx=600.0, fy=600.0, cx=320.0, cy=240.0)
    mask = np.zeros((480, 640), dtype=np.uint8)
    mask[150:350, 300:360] = 1

    # Глубина 2.5 метра — вне досягаемости манипулятора
    depth_map = np.full((480, 640), 2.50, dtype=np.float32)

    target = compute_cut_target(
        mask=mask,
        depth_map=depth_map,
        intrinsics=cam,
        min_distance_m=0.2,
        max_distance_m=1.2,
    )

    assert target is not None
    assert not target.safe_to_cut
