"""Диспетчеризация и координация флота автономных роботов-комбайнов (sprint-27).

Управление группой роботов на винограднике:
1. Распределение рядов (Row Allocation):
   - Оптимальное назначение рядов комбайнам с учётом остатка батареи (%),
     наполненности бункера (кг), прогнозируемой урожайности ряда и уклона склона.
2. Координация разворотных зон (Headlands / поворотные полосы):
   - Предотвращение коллизий и тупиков (deadlocks) в узких междурядьях и на концах рядов;
   - Приоритет выезда комбайна с полным бункером.
3. Логистика разгрузки урожая (Bunker Discharge):
   - Прогнозирование момента заполнения бункера (достижение 90 % объёма);
   - Расчёт точки встречи (rendezvous waypoint) с прицепным трактором-накопителем
     на краю поля без холостого пробега комбайна.
4. Мониторинг телеметрии флота и расчёт совокупного ETA завершения сбора на участке.
"""

import enum
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any


class RobotStatus(enum.StrEnum):
    """Текущий эксплуатационный статус робота в поле."""

    IDLE = "idle"  # Свободен / ожидает команды
    NAVIGATING = "navigating"  # Переход к началу ряда
    HARVESTING = "harvesting"  # Рабочий проезд и срез гроздей
    UNLOADING = "unloading"  # Выгрузка винограда в накопитель
    CHARGING = "charging"  # Подзарядка на полевой станции
    MAINTENANCE = "maintenance"  # Техобслуживание / замена ножей
    EMERGENCY = "emergency"  # Аварийный останов сенсорами


class RowHarvestStatus(enum.StrEnum):
    """Статус уборки шпалерного ряда."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    SKIPPED_BLOCKED = "skipped_blocked"


@dataclass
class RobotAgent:
    """Модель отдельного робота-комбайна в составе флота."""

    robot_id: str
    battery_pct: float
    current_speed_mps: float
    bunker_capacity_kg: float = 350.0
    current_bunker_kg: float = 0.0
    status: RobotStatus = RobotStatus.IDLE
    current_row_id: str | None = None
    lat: float = 47.14
    lon: float = 28.86
    total_harvested_kg: float = 0.0

    @property
    def bunker_fullness_pct(self) -> float:
        return (self.current_bunker_kg / max(1.0, self.bunker_capacity_kg)) * 100.0

    @property
    def is_available(self) -> bool:
        return (
            self.status in (RobotStatus.IDLE, RobotStatus.NAVIGATING)
            and self.battery_pct > 15.0
            and self.bunker_fullness_pct < 90.0
        )


@dataclass
class VineyardRow:
    """Характеристики шпалерного ряда виноградника."""

    row_id: str
    row_number: int
    length_m: float
    start_point: tuple[float, float]  # (lat, lon)
    end_point: tuple[float, float]  # (lat, lon)
    estimated_yield_kg: float
    slope_pct: float = 0.0  # Уклон склона (0..15 %)
    status: RowHarvestStatus = RowHarvestStatus.PENDING
    assigned_robot_id: str | None = None


@dataclass(frozen=True)
class RendezvousEvent:
    """План встречи комбайна с трактором-накопителем для перегрузки урожая."""

    robot_id: str
    rendezvous_location: tuple[float, float]
    predicted_discharge_kg: float
    estimated_arrival_sec: float
    headland_side: str  # "start" | "end"


@dataclass(frozen=True)
class RowAssignmentVerdict:
    """Решение диспетчера о назначении комбайна на конкретный ряд."""

    robot_id: str
    row_id: str
    estimated_time_min: float
    expected_harvest_kg: float
    requires_midway_discharge: bool
    discharge_plan: RendezvousEvent | None = None


@dataclass
class HeadlandOccupancy:
    """Учёт занятости разворотных полос во избежание тупиков и лобовых коллизий."""

    active_turns: dict[str, str] = field(default_factory=dict)  # zone_id -> robot_id

    def request_turn(self, zone_id: str, robot_id: str) -> bool:
        if zone_id in self.active_turns:
            return self.active_turns[zone_id] == robot_id
        self.active_turns[zone_id] = robot_id
        return True

    def release_turn(self, zone_id: str, robot_id: str) -> None:
        if self.active_turns.get(zone_id) == robot_id:
            self.active_turns.pop(zone_id, None)


class FleetDispatcher:
    """Алгоритмический координатор и планировщик флота роботов."""

    def __init__(
        self,
        nominal_harvest_speed_mps: float = 0.35,
        nominal_turn_time_sec: float = 45.0,
        unloading_speed_kg_per_sec: float = 2.5,
        battery_consumption_pct_per_km: float = 12.0,
    ) -> None:
        self.nominal_harvest_speed_mps = nominal_harvest_speed_mps
        self.nominal_turn_time_sec = nominal_turn_time_sec
        self.unloading_speed_kg_per_sec = unloading_speed_kg_per_sec
        self.battery_consumption_pct_per_km = battery_consumption_pct_per_km
        self.headland_manager = HeadlandOccupancy()

    def plan_field_dispatch(
        self,
        robots: Sequence[RobotAgent],
        rows: Sequence[VineyardRow],
    ) -> list[RowAssignmentVerdict]:
        """Оптимально распределяет ряды между комбайнами.

        Жадный алгоритм с учётом доступной ёмкости бункера и минимизации
        холостых переходов между смежными рядами.
        """
        pending_rows = [r for r in rows if r.status == RowHarvestStatus.PENDING]
        available_robots = [r for r in robots if r.is_available]

        assignments: list[RowAssignmentVerdict] = []
        if not pending_rows or not available_robots:
            return assignments

        # Сортируем роботов по наибольшему остатку бункера и батареи
        sorted_robots = sorted(
            available_robots,
            key=lambda r: (
                r.bunker_capacity_kg - r.current_bunker_kg,
                r.battery_pct,
            ),
            reverse=True,
        )

        for row in pending_rows:
            if not sorted_robots:
                break

            # Выбираем наиболее подходящего робота
            best_robot = None
            best_score = -999999.0

            for robot in sorted_robots:
                remaining_bunker = robot.bunker_capacity_kg - robot.current_bunker_kg
                bunker_slack = remaining_bunker - row.estimated_yield_kg

                # Проверка расхода батареи на проезд ряда с учётом уклона
                distance_km = row.length_m / 1000.0
                slope_penalty = 1.0 + (max(0.0, row.slope_pct) / 10.0)
                battery_needed = (
                    distance_km * self.battery_consumption_pct_per_km * slope_penalty
                )

                if robot.battery_pct - battery_needed < 10.0:
                    continue  # Опасность разряда в ряду

                # Расчёт эвристического веса соответствия
                score = (
                    robot.battery_pct * 0.4
                    + remaining_bunker * 0.6
                    - abs(bunker_slack) * 0.2
                )
                if score > best_score:
                    best_score = score
                    best_robot = robot

            if best_robot is None:
                continue

            # Расчёт времени прохождения
            effective_speed = max(
                0.15,
                self.nominal_harvest_speed_mps * (1.0 - row.slope_pct / 30.0),
            )
            traverse_time_sec = (
                row.length_m / effective_speed
            ) + self.nominal_turn_time_sec

            needs_midway_discharge = (
                best_robot.current_bunker_kg + row.estimated_yield_kg
                >= best_robot.bunker_capacity_kg * 0.95
            )

            discharge_event: RendezvousEvent | None = None
            if needs_midway_discharge:
                discharge_event = RendezvousEvent(
                    robot_id=best_robot.robot_id,
                    rendezvous_location=row.end_point,
                    predicted_discharge_kg=round(
                        best_robot.current_bunker_kg + row.estimated_yield_kg,
                        1,
                    ),
                    estimated_arrival_sec=round(traverse_time_sec, 1),
                    headland_side="end",
                )

            verdict = RowAssignmentVerdict(
                robot_id=best_robot.robot_id,
                row_id=row.row_id,
                estimated_time_min=round(traverse_time_sec / 60.0, 1),
                expected_harvest_kg=row.estimated_yield_kg,
                requires_midway_discharge=needs_midway_discharge,
                discharge_plan=discharge_event,
            )
            assignments.append(verdict)

            # Обновляем состояние робота на время симуляции
            best_robot.current_bunker_kg += row.estimated_yield_kg
            row.assigned_robot_id = best_robot.robot_id
            row.status = RowHarvestStatus.IN_PROGRESS

            if best_robot.current_bunker_kg >= best_robot.bunker_capacity_kg * 0.90:
                # Робот исчерпал бункер, снимаем с текущего пула назначений
                sorted_robots.remove(best_robot)

        return assignments

    def calculate_fleet_progress(
        self,
        robots: Sequence[RobotAgent],
        rows: Sequence[VineyardRow],
    ) -> dict[str, Any]:
        """Возвращает сводную статистику прогресса уборочной кампании."""
        total_rows = len(rows)
        completed_rows = sum(1 for r in rows if r.status == RowHarvestStatus.COMPLETED)
        in_progress_rows = sum(
            1 for r in rows if r.status == RowHarvestStatus.IN_PROGRESS
        )
        total_yield_kg = sum(r.estimated_yield_kg for r in rows)
        harvested_so_far_kg = sum(r.total_harvested_kg for r in robots)

        # Совокупная скорость уборки (кг/ч)
        active_robots = [r for r in robots if r.status == RobotStatus.HARVESTING]
        active_throughput_kg_h = sum(
            (r.current_speed_mps * 3600.0)
            * 0.4  # средняя линейная плотность гроздей ~0.4 кг/м
            for r in active_robots
        )

        remaining_kg = max(0.0, total_yield_kg - harvested_so_far_kg)
        eta_hours = (
            remaining_kg / max(10.0, active_throughput_kg_h)
            if active_throughput_kg_h > 0
            else 0.0
        )

        return {
            "total_rows": total_rows,
            "completed_rows": completed_rows,
            "in_progress_rows": in_progress_rows,
            "pending_rows": total_rows - completed_rows - in_progress_rows,
            "progress_pct": round((completed_rows / max(1, total_rows)) * 100.0, 1),
            "total_yield_kg": round(total_yield_kg, 1),
            "harvested_so_far_kg": round(harvested_so_far_kg, 1),
            "active_harvesters_count": len(active_robots),
            "estimated_finish_hours": round(eta_hours, 1),
        }
