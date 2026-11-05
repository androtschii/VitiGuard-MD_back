"""Тесты бортового конвейера инференса реального времени (ml/sprint-26)."""

import numpy as np

from vitiguard_ml.robot.cutting import CameraIntrinsics
from vitiguard_ml.robot.edge_infer import (
    EdgeInferencePipeline,
    HarvestDecision,
)
from vitiguard_ml.robot.kinematics import (
    TrunkObstacle,
)


def _make_mock_frame(
    h: int = 480,
    w: int = 640,
    bunch_y: int = 200,
    bunch_x: int = 300,
    bunch_r: int = 40,
    color_rgb: tuple[int, int, int] = (40, 10, 20),  # Спелый тёмно-красный виноград
    depth_val: float = 0.55,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rgb = np.full((h, w, 3), 120, dtype=np.uint8)
    # Фоновые зелёные листья шпалеры
    rgb[..., 1] = 160

    mask = np.zeros((h, w), dtype=np.uint8)
    yy, xx = np.ogrid[:h, :w]
    circle = (yy - bunch_y) ** 2 + (xx - bunch_x) ** 2 <= bunch_r**2
    mask[circle] = 1
    rgb[circle] = color_rgb

    depth_m = np.full((h, w), depth_val, dtype=np.float32)
    return rgb, depth_m, mask


def test_pipeline_sensor_sanity_checks() -> None:
    pipeline = EdgeInferencePipeline()
    rgb, depth_m, mask = _make_mock_frame()

    # 1. Некорректная размерность RGB (2D вместо 3D)
    res_bad_rgb = pipeline.process_frame(
        rgb=rgb[..., 0], depth_m=depth_m, bunch_masks=[mask]
    )
    assert res_bad_rgb.emergency_stop
    assert "RGB" in str(res_bad_rgb.emergency_stop_reason)

    # 2. Несовпадение габаритов Depth-карты
    res_bad_depth = pipeline.process_frame(
        rgb=rgb, depth_m=depth_m[:200, :200], bunch_masks=[mask]
    )
    assert res_bad_depth.emergency_stop
    assert "Несовпадение габаритов" in str(res_bad_depth.emergency_stop_reason)

    # 3. Отвал датчика глубины (все нули)
    res_dead_depth = pipeline.process_frame(
        rgb=rgb, depth_m=np.zeros_like(depth_m), bunch_masks=[mask]
    )
    assert res_dead_depth.emergency_stop
    assert "Сбой лидара" in str(res_dead_depth.emergency_stop_reason)


def test_pipeline_harvest_candidate_ready() -> None:
    pipeline = EdgeInferencePipeline(
        variety="cabernet_sauvignon",
        min_sugar_brix=18.0,
        max_rot_fraction=0.15,
        rot_matrix=np.eye(3, dtype=np.float64),
        trans_vector=np.array([[0.0], [0.0], [0.0]], dtype=np.float64),
    )
    # Зрелая здоровая гроздь (тёмно-фиолетовая) на рабочей дистанции 0.50 м
    rgb, depth_m, mask = _make_mock_frame(
        bunch_y=240, bunch_x=320, color_rgb=(40, 10, 20), depth_val=0.50
    )

    res = pipeline.process_frame(rgb, depth_m, [mask], frame_id="frame_001")
    assert not res.emergency_stop
    assert res.bunches_count == 1
    assert len(res.harvest_candidates) == 1
    assert len(res.skipped_bunches) == 0

    verdict = res.harvest_candidates[0]
    assert verdict.decision == HarvestDecision.HARVEST
    assert verdict.trajectory is not None
    assert verdict.trajectory.is_collision_free
    assert verdict.trajectory.is_reachable
    assert len(verdict.trajectory.waypoints) == 4


def test_pipeline_skip_infected_bunch() -> None:
    pipeline = EdgeInferencePipeline(max_rot_fraction=0.10)
    # Буро-коричневая окраска некроза
    rgb, depth_m, mask = _make_mock_frame(color_rgb=(120, 70, 50), depth_val=0.50)

    res = pipeline.process_frame(rgb, depth_m, [mask])
    assert not res.emergency_stop
    assert len(res.harvest_candidates) == 0
    assert len(res.skipped_bunches) == 1

    verdict = res.skipped_bunches[0]
    assert verdict.decision == HarvestDecision.SKIP_INFECTED
    assert "серой гнилью" in verdict.decision_reason


def test_pipeline_skip_unripe_bunch() -> None:
    pipeline = EdgeInferencePipeline(variety="cabernet_sauvignon", min_sugar_brix=19.5)
    # Ярко-зелёные незрелые ягоды (хлорофилл, сахаристость ~11-13 Brix)
    rgb, depth_m, mask = _make_mock_frame(color_rgb=(50, 160, 60), depth_val=0.50)

    res = pipeline.process_frame(rgb, depth_m, [mask])
    assert not res.emergency_stop
    assert len(res.harvest_candidates) == 0
    assert len(res.skipped_bunches) == 1

    verdict = res.skipped_bunches[0]
    assert verdict.decision == HarvestDecision.SKIP_UNRIPE
    assert "сахаристость" in verdict.decision_reason


def test_pipeline_skip_obstacle_collision() -> None:
    intrinsics = CameraIntrinsics(fx=500.0, fy=500.0, cx=320.0, cy=240.0)
    pipeline = EdgeInferencePipeline(
        intrinsics=intrinsics,
        min_sugar_brix=16.0,
        rot_matrix=np.eye(3, dtype=np.float64),
        trans_vector=np.zeros((3, 1), dtype=np.float64),
    )
    rgb, depth_m, mask = _make_mock_frame(
        bunch_y=240, bunch_x=320, color_rgb=(40, 10, 20), depth_val=0.50
    )

    # Ставим ствол прямо на пути секатора в X=0.0, Y=0.0, Z=0.48
    obstacle = TrunkObstacle(
        x=0.0,
        y=0.0,
        z=0.48,
        radius_m=0.10,
        height_m=1.0,
    )

    res = pipeline.process_frame(rgb, depth_m, [mask], trunk_obstacles=[obstacle])
    assert not res.emergency_stop
    assert len(res.harvest_candidates) == 0
    assert len(res.skipped_bunches) == 1
    assert res.skipped_bunches[0].decision == HarvestDecision.SKIP_OBSTACLE


def test_pipeline_skip_unreachable() -> None:
    pipeline = EdgeInferencePipeline(
        min_sugar_brix=16.0,
        rot_matrix=np.eye(3, dtype=np.float64),
        trans_vector=np.zeros((3, 1), dtype=np.float64),
    )
    # Глубина 1.05 м — в поле зрения камеры (<= 1.20 м), но за пределами досягаемости руки (MAX_REACH_RADIUS_M = 0.95 м)
    rgb, depth_m, mask = _make_mock_frame(
        bunch_y=240, bunch_x=320, color_rgb=(40, 10, 20), depth_val=1.05
    )

    res = pipeline.process_frame(rgb, depth_m, [mask])
    assert not res.emergency_stop
    assert len(res.harvest_candidates) == 0
    assert len(res.skipped_bunches) == 1
    assert res.skipped_bunches[0].decision == HarvestDecision.SKIP_UNREACHABLE


def test_pipeline_benchmark() -> None:
    pipeline = EdgeInferencePipeline()
    rgb, depth_m, mask = _make_mock_frame(depth_val=0.50)

    bench = pipeline.benchmark(rgb, depth_m, [mask], num_runs=5)
    assert bench["mean_ms"] > 0
    assert bench["p50_ms"] > 0
    assert bench["fps"] > 0
    assert pipeline.metrics.frame_count == 5

    pcts = pipeline.metrics.percentiles()
    assert "p50" in pcts
    assert "p95" in pcts
    assert "p99" in pcts
