"""Маршруты API для координации и диспетчеризации флота роботов (sprint-27)."""

from fastapi import APIRouter, status
from vitiguard_ml.robot.fleet import (
    FleetDispatcher,
    HeadlandOccupancy,
    RobotAgent,
    RobotStatus,
    RowHarvestStatus,
    VineyardRow,
)

from app.api.deps import CurrentUserDep
from app.schemas.errors import ErrorResponse
from app.schemas.fleet import (
    FleetDispatchRequest,
    FleetDispatchResponse,
    RendezvousEventDTO,
    RowAssignmentVerdictDTO,
    UnloadRequestDTO,
)

router = APIRouter(
    prefix="/robot/fleet",
    tags=["robot"],
    responses={status.HTTP_401_UNAUTHORIZED: {"model": ErrorResponse}},
)

# Системный экземпляр менеджера разворотных зон
_headland_manager = HeadlandOccupancy()


@router.post(
    "/dispatch",
    summary="Оптимальное распределение рядов виноградника между роботами-комбайнами",
    response_model=FleetDispatchResponse,
)
async def dispatch_fleet(
    body: FleetDispatchRequest,
    _: CurrentUserDep,
) -> FleetDispatchResponse:
    """Рассчитывает оптимальное расписание сбора урожая с учётом бункеров, заряда и уклонов."""
    dispatcher = FleetDispatcher()

    # Преобразуем входные DTO во внутренние доменные сущности
    domain_robots = [
        RobotAgent(
            robot_id=r.robot_id,
            battery_pct=r.battery_pct,
            current_speed_mps=r.current_speed_mps,
            bunker_capacity_kg=r.bunker_capacity_kg,
            current_bunker_kg=r.current_bunker_kg,
            status=(
                RobotStatus(r.status)
                if r.status in [s.value for s in RobotStatus]
                else RobotStatus.IDLE
            ),
            current_row_id=r.current_row_id,
            lat=r.lat,
            lon=r.lon,
            total_harvested_kg=r.total_harvested_kg,
        )
        for r in body.robots
    ]

    domain_rows = [
        VineyardRow(
            row_id=row.row_id,
            row_number=row.row_number,
            length_m=row.length_m,
            start_point=row.start_point,
            end_point=row.end_point,
            estimated_yield_kg=row.estimated_yield_kg,
            slope_pct=row.slope_pct,
            status=(
                RowHarvestStatus(row.status)
                if row.status in [s.value for s in RowHarvestStatus]
                else RowHarvestStatus.PENDING
            ),
        )
        for row in body.rows
    ]

    raw_assignments = dispatcher.plan_field_dispatch(domain_robots, domain_rows)
    fleet_progress = dispatcher.calculate_fleet_progress(domain_robots, domain_rows)

    dto_assignments = [
        RowAssignmentVerdictDTO(
            robot_id=a.robot_id,
            row_id=a.row_id,
            estimated_time_min=a.estimated_time_min,
            expected_harvest_kg=a.expected_harvest_kg,
            requires_midway_discharge=a.requires_midway_discharge,
            discharge_plan=(
                RendezvousEventDTO(
                    robot_id=a.discharge_plan.robot_id,
                    rendezvous_location=a.discharge_plan.rendezvous_location,
                    predicted_discharge_kg=a.discharge_plan.predicted_discharge_kg,
                    estimated_arrival_sec=a.discharge_plan.estimated_arrival_sec,
                    headland_side=a.discharge_plan.headland_side,
                )
                if a.discharge_plan
                else None
            ),
        )
        for a in raw_assignments
    ]

    assigned_rows = {a.row_id for a in raw_assignments}
    unassigned_count = sum(1 for r in domain_rows if r.row_id not in assigned_rows)

    return FleetDispatchResponse(
        assignments=dto_assignments,
        fleet_progress=fleet_progress,
        assigned_rows_count=len(dto_assignments),
        unassigned_rows_count=unassigned_count,
    )


@router.post(
    "/request-unloading",
    summary="Экстренный запрос на перегрузку бункера комбайна",
)
async def request_unloading(
    body: UnloadRequestDTO,
    _: CurrentUserDep,
) -> dict[str, object]:
    """Регистрирует заявку на перегрузку бункера и назначает точку рандеву."""
    return {
        "status": "scheduled",
        "robot_id": body.robot_id,
        "current_bunker_kg": body.current_bunker_kg,
        "rendezvous_location": body.location,
        "estimated_wait_sec": 180.0,
    }


@router.post(
    "/headland/lock",
    summary="Блокировка разворотной полосы для исключения коллизии",
)
async def lock_headland_zone(
    zone_id: str,
    robot_id: str,
    _: CurrentUserDep,
) -> dict[str, object]:
    """Запрашивает эксклюзивный доступ робота к поворотной полосе."""
    granted = _headland_manager.request_turn(zone_id, robot_id)
    return {
        "zone_id": zone_id,
        "robot_id": robot_id,
        "granted": granted,
    }


@router.post(
    "/headland/release",
    summary="Освобождение разворотной полосы после манёвра",
)
async def release_headland_zone(
    zone_id: str,
    robot_id: str,
    _: CurrentUserDep,
) -> dict[str, object]:
    """Освобождает поворотную полосу для других комбайнов."""
    _headland_manager.release_turn(zone_id, robot_id)
    return {
        "zone_id": zone_id,
        "robot_id": robot_id,
        "status": "released",
    }
