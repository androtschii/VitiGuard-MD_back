import numpy as np

from vitiguard_ml.robot.kinematics import (
    TrunkObstacle,
    check_collision_with_trunks,
    check_reachability,
    distance_segment_to_point,
    generate_harvest_trajectory,
    transform_camera_to_base,
)


def test_transform_camera_to_base() -> None:
    # Единичная матрица вращения и нулевое смещение
    p_cam = (0.1, 0.2, 0.5)
    rot = np.eye(3)
    trans = np.zeros((3, 1))

    p_base = transform_camera_to_base(p_cam, rot, trans)
    assert np.allclose(p_base, [0.1, 0.2, 0.5])


def test_check_reachability() -> None:
    # В пределах рабочей зоны 0.25..0.95 м
    reachable, dist = check_reachability(
        (0.3, 0.3, 0.2), min_radius=0.25, max_radius=0.95
    )
    assert reachable is True
    assert 0.25 <= dist <= 0.95

    # Слишком далеко (1.5 м)
    too_far, _ = check_reachability((1.0, 1.0, 1.0))
    assert too_far is False

    # Слишком близко (0.1 м)
    too_close, _ = check_reachability((0.05, 0.05, 0.05))
    assert too_close is False


def test_distance_segment_to_point() -> None:
    p1 = np.array([0.0, 0.0, 0.0])
    p2 = np.array([1.0, 0.0, 0.0])

    # Точка на перпендикуляре в середине отрезка на высоте 0.5 м
    pt = np.array([0.5, 0.5, 0.0])
    dist = distance_segment_to_point(p1, p2, pt)
    assert np.isclose(dist, 0.5)

    # Точка за концом отрезка (1.5, 0.0, 0.0) -> расстояние до конца = 0.5 м
    pt2 = np.array([1.5, 0.0, 0.0])
    dist2 = distance_segment_to_point(p1, p2, pt2)
    assert np.isclose(dist2, 0.5)


def test_collision_detection_flags_obstacle() -> None:
    p1 = np.array([0.0, 0.0, 0.5])
    p2 = np.array([0.5, 0.0, 0.5])

    # Ствол прямо на пути движения манипулятора
    trunk_blocking = TrunkObstacle(x=0.25, y=0.02, z=0.5, radius_m=0.06)
    col_free, clearance, warnings = check_collision_with_trunks(
        [p1, p2], [trunk_blocking], gripper_safety_radius=0.05
    )
    assert col_free is False
    assert clearance < 0.0
    assert len(warnings) > 0


def test_generate_harvest_trajectory_valid_path() -> None:
    cut_cam = (0.05, 0.0, 0.65)
    plan = generate_harvest_trajectory(cut_cam)

    assert plan.is_valid is True
    assert plan.is_reachable is True
    assert plan.is_collision_free is True
    assert len(plan.waypoints) == 4
    names = [wp.name for wp in plan.waypoints]
    assert names == ["standoff", "pre_cut", "cut", "retract"]
    assert plan.waypoints[2].gripper_action == "cut_and_grip"
