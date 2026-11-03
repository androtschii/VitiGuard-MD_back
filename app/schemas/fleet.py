"""Pydantic-схемы для флота роботов и диспетчеризации сбора урожая (sprint-27)."""

from typing import Any

from pydantic import BaseModel, Field

from app.schemas.base import RequestSchema


class RobotAgentDTO(BaseModel):
    """Снимок состояния отдельного робота-комбайна."""

    robot_id: str = Field(
        description="Уникальный идентификатор робота", examples=["harvester-01"]
    )
    battery_pct: float = Field(description="Уровень заряда АКБ (%)", examples=[82.5])
    current_speed_mps: float = Field(
        description="Текущая скорость (м/с)", examples=[0.35]
    )
    bunker_capacity_kg: float = Field(default=350.0, description="Ёмкость бункера (кг)")
    current_bunker_kg: float = Field(
        default=0.0, description="Текущая масса ягод в бункере (кг)"
    )
    status: str = Field(
        default="idle", description="Статус: idle, harvesting, unloading, charging"
    )
    current_row_id: str | None = Field(default=None, description="Текущий рабочий ряд")
    lat: float = Field(default=47.14, description="Текущая широта")
    lon: float = Field(default=28.86, description="Текущая долгота")
    total_harvested_kg: float = Field(default=0.0, description="Собрано за смену (кг)")


class VineyardRowDTO(BaseModel):
    """Шпалерный ряд виноградника под уборку."""

    row_id: str = Field(description="Идентификатор ряда", examples=["row_42"])
    row_number: int = Field(description="Номер ряда", examples=[42])
    length_m: float = Field(description="Длина ряда (м)", examples=[250.0])
    start_point: tuple[float, float] = Field(description="(lat, lon) начала ряда")
    end_point: tuple[float, float] = Field(description="(lat, lon) конца ряда")
    estimated_yield_kg: float = Field(
        description="Оценочная урожайность ряда (кг)", examples=[115.0]
    )
    slope_pct: float = Field(default=0.0, description="Уклон склона (%)")
    status: str = Field(
        default="pending", description="Статус: pending, in_progress, completed"
    )


class RendezvousEventDTO(BaseModel):
    """План встречи для перегрузки винограда в трактор."""

    robot_id: str
    rendezvous_location: tuple[float, float]
    predicted_discharge_kg: float
    estimated_arrival_sec: float
    headland_side: str


class RowAssignmentVerdictDTO(BaseModel):
    """Назначение робота на ряд."""

    robot_id: str
    row_id: str
    estimated_time_min: float
    expected_harvest_kg: float
    requires_midway_discharge: bool
    discharge_plan: RendezvousEventDTO | None = None


class FleetDispatchRequest(RequestSchema):
    """Запрос на автоматическое распределение рядов."""

    robots: list[RobotAgentDTO]
    rows: list[VineyardRowDTO]


class FleetDispatchResponse(BaseModel):
    """Сформированный план диспетчеризации флота."""

    assignments: list[RowAssignmentVerdictDTO]
    fleet_progress: dict[str, Any]
    assigned_rows_count: int
    unassigned_rows_count: int


class UnloadRequestDTO(RequestSchema):
    """Ручной запрос на вызов трактора-перегрузчика."""

    robot_id: str
    current_bunker_kg: float
    location: tuple[float, float]
