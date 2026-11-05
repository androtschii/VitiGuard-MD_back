"""Маршруты API для приёма телеметрии датчиков LoRaWAN и предупреждения заморозков/града (sprint-29)."""

from fastapi import APIRouter, status
from pydantic import BaseModel, Field

from app.api.deps import CurrentUserDep
from app.schemas.errors import ErrorResponse
from app.schemas.sensor_mesh import (
    FrostRiskReportDTO,
    HailStormRiskReportDTO,
    SensorTelemetryDTO,
    VineyardMeshStatusResponse,
)
from app.services.sensor_mesh import (
    build_mesh_status,
    evaluate_frost_risk,
    evaluate_hail_risk,
)

router = APIRouter(
    prefix="/sensors",
    tags=["sensors"],
    responses={status.HTTP_401_UNAUTHORIZED: {"model": ErrorResponse}},
)


class MeshStatusRequest(BaseModel):
    """Показания датчиков сети участка для проверки её состояния."""

    vineyard_id: str
    sensors: list[SensorTelemetryDTO]


class FrostAssessmentRequest(BaseModel):
    """Показания датчиков и фаза развития лозы для оценки опасности заморозка."""

    vineyard_id: str
    sensors: list[SensorTelemetryDTO]
    phenology_stage: str = Field(default="BBCH 09 (Распускание почек)")


class HailAssessmentRequest(BaseModel):
    """Показания датчика и падение давления для оценки опасности града."""

    vineyard_id: str
    telemetry: SensorTelemetryDTO
    pressure_drop_1h_hpa: float = Field(
        default=0.0, description="Скорость падения давления за последний час (гПа/ч)"
    )


@router.post(
    "/telemetry",
    summary="Приём телеметрии от полевого узла датчиков LoRaWAN",
)
async def submit_sensor_telemetry(
    body: SensorTelemetryDTO,
    _: CurrentUserDep,
) -> dict[str, object]:
    """Принимает пакет измерений микроклимата кроны и влажности почвы от полевого зонда."""
    return {
        "status": "recorded",
        "sensor_id": body.sensor_id,
        "vineyard_id": body.vineyard_id,
        "timestamp": body.timestamp.isoformat(),
        "canopy_temperature_c": body.canopy_temperature_c,
    }


@router.post(
    "/mesh-status",
    summary="Получение сводного статуса сенсорной сети и интерполяционной карты",
    response_model=VineyardMeshStatusResponse,
)
async def get_mesh_status(
    body: MeshStatusRequest,
    _: CurrentUserDep,
) -> VineyardMeshStatusResponse:
    """Агрегирует данные всех узлов виноградника и строит пространственную сетку температур."""
    return build_mesh_status(body.vineyard_id, body.sensors)


@router.post(
    "/frost-risk",
    summary="Оценка угрозы весенних заморозков для почек",
    response_model=FrostRiskReportDTO,
)
async def assess_frost_risk(
    body: FrostAssessmentRequest,
    _: CurrentUserDep,
) -> FrostRiskReportDTO:
    """Оценивает риск заморозков с учётом инверсии, морозных карманов и влажного термометра."""
    return evaluate_frost_risk(body.vineyard_id, body.sensors, body.phenology_stage)


@router.post(
    "/hail-risk",
    summary="Оценка риска конвективного града и необходимости закрытия сеток",
    response_model=HailStormRiskReportDTO,
)
async def assess_hail_risk(
    body: HailAssessmentRequest,
    _: CurrentUserDep,
) -> HailStormRiskReportDTO:
    """Определяет риск града по барометрическому градиенту и конвективному индексу."""
    return evaluate_hail_risk(
        body.vineyard_id, body.telemetry, body.pressure_drop_1h_hpa
    )
