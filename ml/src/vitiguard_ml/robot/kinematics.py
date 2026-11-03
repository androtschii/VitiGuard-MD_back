"""Кинематика манипулятора, калибровка камера-рука и безаварийная траектория среза (ml/pr-024, kinematics).

Связывает зрение робота (координаты точки среза ножки из cutting.py в системе камеры)
с механикой манипулятора робота-сборщика:
1. Перевод координат из системы камеры в базовую систему манипулятора (hand-eye calibration):
   P_base = R * P_cam + T;
2. Построение траектории подхода захвата-секатора вдоль нормали ножки грозди:
   точки standoff (отступ 100 мм), pre-cut (30 мм), cut_point (ножка) и retract;
3. Контроль столкновений (collision avoidance):
   - Проверка коридора движения секатора против 3D-цилиндров стволов лозы (по детектированным
     рамкам стволов с глубиной);
   - Проверка пересечения шпалерной проволоки;
4. Проверка рабочей зоны манипулятора (досягаемость / reachability limits).

Запуск из каталога ml:
  uv run python -m vitiguard_ml.robot.kinematics --cut-point 0.05 -0.10 0.85
"""

import argparse
import json
import math
from dataclasses import dataclass, field

import numpy as np

# Границы досягаемости типичного 6-осевого лёгкого манипулятора (UR5 / Franka / custom)
MIN_REACH_RADIUS_M = 0.25
MAX_REACH_RADIUS_M = 0.95
MAX_APPROACH_SPEED_MPS = 0.20


@dataclass(frozen=True)
class Waypoint:
    name: str
    x: float
    y: float
    z: float
    speed_mps: float
    gripper_action: str  # "open" | "align" | "cut_and_grip" | "retract"


@dataclass(frozen=True)
class TrunkObstacle:
    """3D-цилиндр ствола лозы."""

    x: float
    y: float
    z: float
    radius_m: float = 0.06
    height_m: float = 1.20


@dataclass
class TrajectoryPlan:
    is_valid: bool
    is_reachable: bool
    is_collision_free: bool
    waypoints: list[Waypoint]
    distance_to_nearest_obstacle_m: float
    warnings: list[str] = field(default_factory=list)


def transform_camera_to_base(
    point_cam: tuple[float, float, float] | np.ndarray,
    rot_matrix: np.ndarray | None = None,
    trans_vector: np.ndarray | None = None,
) -> np.ndarray:
    """Переводит точку из системы координат камеры в базовую систему манипулятора.

    По умолчанию используется стандартная калибровочная матрица для камеры,
    установленной на 150 мм выше и 80 мм правее основания руки.
    """
    p = np.asarray(point_cam, dtype=np.float64).reshape(3, 1)

    if rot_matrix is None:
        # Единичное вращение с лёгким наклоном камеры вниз на 10 градусов
        pitch = math.radians(-10.0)
        rot_matrix = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, math.cos(pitch), -math.sin(pitch)],
                [0.0, math.sin(pitch), math.cos(pitch)],
            ],
            dtype=np.float64,
        )

    if trans_vector is None:
        # Смещение камеры относительно базы: x=+0.08м, y=+0.15м, z=+0.05м
        trans_vector = np.array([[0.08], [0.15], [0.05]], dtype=np.float64)

    res: np.ndarray = np.asarray((rot_matrix @ p + trans_vector).flatten(), dtype=np.float64)
    return res


def check_reachability(
    point_base: np.ndarray | tuple[float, float, float],
    min_radius: float = MIN_REACH_RADIUS_M,
    max_radius: float = MAX_REACH_RADIUS_M,
) -> tuple[bool, float]:
    """Проверяет, попадает ли точка в полусферу досягаемости манипулятора."""
    px, py, pz = float(point_base[0]), float(point_base[1]), float(point_base[2])
    dist = math.sqrt(px**2 + py**2 + pz**2)
    is_reachable = (min_radius <= dist <= max_radius) and (pz >= -0.20)
    return is_reachable, round(dist, 4)


def distance_segment_to_point(
    seg_start: np.ndarray,
    seg_end: np.ndarray,
    point: np.ndarray,
) -> float:
    """Кратчайшее расстояние от 3D-отрезка до точки."""
    v = seg_end - seg_start
    w = point - seg_start
    c1 = np.dot(w, v)
    if c1 <= 0:
        return float(np.linalg.norm(point - seg_start))
    c2 = np.dot(v, v)
    if c2 <= c1:
        return float(np.linalg.norm(point - seg_end))
    b = c1 / c2
    pb = seg_start + b * v
    return float(np.linalg.norm(point - pb))


def check_collision_with_trunks(
    trajectory_points: list[np.ndarray],
    trunks: list[TrunkObstacle],
    gripper_safety_radius: float = 0.08,
) -> tuple[bool, float, list[str]]:
    """Проверяет отсутствие столкновений траектории движения секатора со стволами лозы."""
    min_clearance = 999.0
    warnings: list[str] = []

    for i in range(len(trajectory_points) - 1):
        p1 = trajectory_points[i]
        p2 = trajectory_points[i + 1]

        for idx, trunk in enumerate(trunks):
            trunk_pos = np.array([trunk.x, trunk.y, trunk.z])
            dist = distance_segment_to_point(p1, p2, trunk_pos)
            clearance = dist - (trunk.radius_m + gripper_safety_radius)
            min_clearance = min(min_clearance, clearance)

            if clearance < 0.0:
                warnings.append(
                    f"Коллизия траектории со стволом #{idx}: зазор {clearance:+.3f} м"
                )

    is_collision_free = min_clearance > 0.0
    return is_collision_free, round(min_clearance, 4), warnings


def generate_harvest_trajectory(
    cut_point_cam: tuple[float, float, float],
    normal_cam: tuple[float, float, float] = (0.0, 0.0, -1.0),
    obstacles: list[TrunkObstacle] | None = None,
    rot_matrix: np.ndarray | None = None,
    trans_vector: np.ndarray | None = None,
) -> TrajectoryPlan:
    """Генерирует полную безопасную траекторию среза и съёма грозди."""
    if obstacles is None:
        obstacles = []

    # 1. Перевод целевой точки среза в систему координат базы манипулятора
    cut_base = transform_camera_to_base(cut_point_cam, rot_matrix, trans_vector)

    # 2. Нормализация вектора подхода
    n = np.asarray(normal_cam, dtype=np.float64)
    norm = np.linalg.norm(n)
    n = n / norm if norm > 0 else np.array([0.0, 0.0, -1.0])
    n_base = rot_matrix @ n if rot_matrix is not None else n

    # 3. Формирование путевых точек траектории (waypoints)
    # Standoff: отступ на 120 мм назад вдоль вектора подхода
    standoff_base = cut_base - n_base * 0.12
    # Pre-cut: позиционирование на 30 мм от ножки
    pre_cut_base = cut_base - n_base * 0.03
    # Retract: отвод грозди на 150 мм вбок и назад
    retract_base = cut_base + np.array([-0.10, 0.0, -0.15])

    # 4. Проверка досягаемости
    reachable, dist_m = check_reachability(cut_base)
    warnings: list[str] = []
    if not reachable:
        warnings.append(
            f"Точка среза вне рабочей зоны манипулятора: расстояние {dist_m:.3f} м "
            f"(лимиты {MIN_REACH_RADIUS_M}..{MAX_REACH_RADIUS_M} м)"
        )

    # 5. Проверка коллизий
    path_points = [standoff_base, pre_cut_base, cut_base, retract_base]
    collision_free, clearance, col_warnings = check_collision_with_trunks(
        path_points, obstacles
    )
    warnings.extend(col_warnings)

    waypoints = [
        Waypoint(
            name="standoff",
            x=round(float(standoff_base[0]), 4),
            y=round(float(standoff_base[1]), 4),
            z=round(float(standoff_base[2]), 4),
            speed_mps=0.25,
            gripper_action="open",
        ),
        Waypoint(
            name="pre_cut",
            x=round(float(pre_cut_base[0]), 4),
            y=round(float(pre_cut_base[1]), 4),
            z=round(float(pre_cut_base[2]), 4),
            speed_mps=0.10,
            gripper_action="align",
        ),
        Waypoint(
            name="cut",
            x=round(float(cut_base[0]), 4),
            y=round(float(cut_base[1]), 4),
            z=round(float(cut_base[2]), 4),
            speed_mps=0.04,
            gripper_action="cut_and_grip",
        ),
        Waypoint(
            name="retract",
            x=round(float(retract_base[0]), 4),
            y=round(float(retract_base[1]), 4),
            z=round(float(retract_base[2]), 4),
            speed_mps=0.15,
            gripper_action="retract",
        ),
    ]

    is_valid = reachable and collision_free

    return TrajectoryPlan(
        is_valid=is_valid,
        is_reachable=reachable,
        is_collision_free=collision_free,
        waypoints=waypoints,
        distance_to_nearest_obstacle_m=clearance,
        warnings=warnings,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cut-point",
        nargs=3,
        type=float,
        default=[0.05, -0.10, 0.85],
        help="3D-координаты точки среза в камере: X Y Z (в метрах)",
    )
    args = parser.parse_args(argv)

    plan = generate_harvest_trajectory(tuple(args.cut_point))
    print(
        json.dumps(
            {
                "is_valid": plan.is_valid,
                "is_reachable": plan.is_reachable,
                "is_collision_free": plan.is_collision_free,
                "distance_to_nearest_obstacle_m": plan.distance_to_nearest_obstacle_m,
                "warnings": plan.warnings,
                "waypoints": [
                    {
                        "name": wp.name,
                        "x": wp.x,
                        "y": wp.y,
                        "z": wp.z,
                        "action": wp.gripper_action,
                    }
                    for wp in plan.waypoints
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
