"""Тесты планировщика и координатора флота роботов (sprint-27)."""

import pytest

from app.services.fleet_dispatch import (
    FleetDispatcher,
    HeadlandOccupancy,
    RobotAgent,
    RobotStatus,
    RowHarvestStatus,
    VineyardRow,
)


def test_robot_agent_availability() -> None:
    """Проверка доступности робота для назначения рядов."""
    ready_robot = RobotAgent(
        robot_id="harv-01",
        battery_pct=85.0,
        current_speed_mps=0.4,
        bunker_capacity_kg=350.0,
        current_bunker_kg=50.0,
        status=RobotStatus.IDLE,
    )
    assert ready_robot.is_available is True
    assert ready_robot.bunker_fullness_pct == pytest.approx(14.28, rel=1e-2)

    # Заполненный бункер (>90%)
    full_robot = RobotAgent(
        robot_id="harv-02",
        battery_pct=90.0,
        current_speed_mps=0.4,
        bunker_capacity_kg=350.0,
        current_bunker_kg=330.0,  # 94%
        status=RobotStatus.IDLE,
    )
    assert full_robot.is_available is False

    # Разряженная батарея (<=15%)
    dead_robot = RobotAgent(
        robot_id="harv-03",
        battery_pct=12.0,
        current_speed_mps=0.4,
        status=RobotStatus.IDLE,
    )
    assert dead_robot.is_available is False


def test_headland_occupancy_lock_and_release() -> None:
    """Проверка блокировки разворотной зоны для исключения столкновений."""
    headland = HeadlandOccupancy()
    zone = "headland_north_row_12"

    # Первый робот успешно бронирует
    assert headland.request_turn(zone, "robot-1") is True
    # Повторный запрос того же робота разрешён (идемпотентность)
    assert headland.request_turn(zone, "robot-1") is True

    # Второй робот не может занять ту же зону
    assert headland.request_turn(zone, "robot-2") is False

    # Освобождение зоны первым роботом
    headland.release_turn(zone, "robot-1")

    # Теперь второй робот может занять зону
    assert headland.request_turn(zone, "robot-2") is True


def test_fleet_dispatcher_allocation() -> None:
    """Проверка оптимального назначения рядов двум комбайнам."""
    dispatcher = FleetDispatcher(
        nominal_harvest_speed_mps=0.5,
        nominal_turn_time_sec=30.0,
    )

    robots = [
        RobotAgent(
            robot_id="r1",
            battery_pct=90.0,
            current_speed_mps=0.5,
            bunker_capacity_kg=400.0,
            current_bunker_kg=0.0,
        ),
        RobotAgent(
            robot_id="r2",
            battery_pct=80.0,
            current_speed_mps=0.5,
            bunker_capacity_kg=400.0,
            current_bunker_kg=0.0,
        ),
    ]

    rows = [
        VineyardRow(
            row_id="row_01",
            row_number=1,
            length_m=300.0,
            start_point=(47.141, 28.861),
            end_point=(47.144, 28.861),
            estimated_yield_kg=120.0,
            slope_pct=2.0,
        ),
        VineyardRow(
            row_id="row_02",
            row_number=2,
            length_m=300.0,
            start_point=(47.141, 28.862),
            end_point=(47.144, 28.862),
            estimated_yield_kg=130.0,
            slope_pct=3.0,
        ),
    ]

    assignments = dispatcher.plan_field_dispatch(robots, rows)

    assert len(assignments) == 2
    assigned_robots = {a.robot_id for a in assignments}
    assert "r1" in assigned_robots
    assert "r2" in assigned_robots

    # Время прохождения 300м при ~0.5 м/с + 30с разворот ~ 630 сек (~10.5 мин)
    assert 9.0 <= assignments[0].estimated_time_min <= 12.0
    assert assignments[0].expected_harvest_kg in (120.0, 130.0)


def test_rendezvous_trigger_when_bunker_full() -> None:
    """Если бункер почти полон, планируется точка выгрузки (rendezvous)."""
    dispatcher = FleetDispatcher()
    robot = RobotAgent(
        robot_id="r-heavy",
        battery_pct=95.0,
        current_speed_mps=0.4,
        bunker_capacity_kg=200.0,
        current_bunker_kg=150.0,  # + 60 кг урожая превысит 200 * 0.95 = 190 кг
    )
    row = VineyardRow(
        row_id="row_dense",
        row_number=5,
        length_m=200.0,
        start_point=(47.10, 28.80),
        end_point=(47.12, 28.80),
        estimated_yield_kg=60.0,
    )

    assignments = dispatcher.plan_field_dispatch([robot], [row])
    assert len(assignments) == 1
    assert assignments[0].requires_midway_discharge is True
    plan = assignments[0].discharge_plan
    assert plan is not None
    assert plan.predicted_discharge_kg == 210.0
    assert plan.rendezvous_location == (47.12, 28.80)


def test_fleet_progress_calculation() -> None:
    """Проверка расчёта прогресса кампании сбора урожая."""
    dispatcher = FleetDispatcher()
    robots = [
        RobotAgent(
            robot_id="r1",
            battery_pct=80.0,
            current_speed_mps=0.4,
            status=RobotStatus.HARVESTING,
            total_harvested_kg=350.0,
        ),
        RobotAgent(
            robot_id="r2",
            battery_pct=75.0,
            current_speed_mps=0.4,
            status=RobotStatus.HARVESTING,
            total_harvested_kg=450.0,
        ),
    ]

    rows = [
        VineyardRow(
            "r1",
            1,
            200.0,
            (0.0, 0.0),
            (0.0, 0.0),
            200.0,
            status=RowHarvestStatus.COMPLETED,
        ),
        VineyardRow(
            "r2",
            2,
            200.0,
            (0.0, 0.0),
            (0.0, 0.0),
            300.0,
            status=RowHarvestStatus.IN_PROGRESS,
        ),
        VineyardRow(
            "r3",
            3,
            200.0,
            (0.0, 0.0),
            (0.0, 0.0),
            300.0,
            status=RowHarvestStatus.PENDING,
        ),
        VineyardRow(
            "r4",
            4,
            200.0,
            (0.0, 0.0),
            (0.0, 0.0),
            200.0,
            status=RowHarvestStatus.PENDING,
        ),
    ]

    progress = dispatcher.calculate_fleet_progress(robots, rows)
    assert progress["total_rows"] == 4
    assert progress["completed_rows"] == 1
    assert progress["in_progress_rows"] == 1
    assert progress["pending_rows"] == 2
    assert progress["progress_pct"] == 25.0
    assert progress["harvested_so_far_kg"] == 800.0
    assert progress["active_harvesters_count"] == 2
    assert progress["estimated_finish_hours"] > 0
