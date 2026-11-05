"""Маршруты API для ультравысокоточного БПЛА-анализа и инвентаризации лозы (sprint-30)."""

from fastapi import APIRouter, status

from app.api.deps import CurrentUserDep
from app.schemas.errors import ErrorResponse
from app.schemas.uav import (
    MultispectralPixelDTO,
    ReplantingOrderRequest,
    ReplantingOrderResponse,
    UAVAnalysisRequest,
    UAVAnalysisResponse,
)
from app.services.uav import (
    analyze_uav_transects,
    generate_replanting_specification,
)

router = APIRouter(
    prefix="/uav",
    tags=["uav"],
    responses={status.HTTP_401_UNAUTHORIZED: {"model": ErrorResponse}},
)


@router.post(
    "/analyze-transects",
    summary="Анализ мультиспектральных трансектов БПЛА (детекция выпадов и стресса)",
    response_model=UAVAnalysisResponse,
)
async def analyze_transects(
    body: UAVAnalysisRequest,
    _: CurrentUserDep,
) -> UAVAnalysisResponse:
    """Определяет пропуски в шпалере (> 1.2 м), координаты выпавших кустов и очаги хлороза по NDRE."""
    return analyze_uav_transects(body)


@router.post(
    "/replanting-order",
    summary="Формирование ведомости подсадки саженцев для питомника",
    response_model=ReplantingOrderResponse,
)
async def replanting_order(
    body: ReplantingOrderRequest,
    _: CurrentUserDep,
) -> ReplantingOrderResponse:
    """Рассчитывает процент выпада кустов, страховой резерв (+5%) и экспортирует GPS-точки для бригады."""
    return generate_replanting_specification(body)


@router.post(
    "/indices-preview",
    summary="Экспресс-расчёт индексов NDVI, NDRE и GNDVI для контрольной точки",
)
async def preview_indices(
    pixel: MultispectralPixelDTO,
    _: CurrentUserDep,
) -> dict[str, float]:
    """Быстрый калькулятор вегетационных индексов по 4 мультиспектральным каналам."""
    ndvi = (pixel.nir - pixel.red) / max(1e-4, pixel.nir + pixel.red)
    ndre = (pixel.nir - pixel.red_edge) / max(1e-4, pixel.nir + pixel.red_edge)
    gndvi = (pixel.nir - pixel.green) / max(1e-4, pixel.nir + pixel.green)

    return {
        "ndvi": round(ndvi, 3),
        "ndre": round(ndre, 3),
        "gndvi": round(gndvi, 3),
    }
