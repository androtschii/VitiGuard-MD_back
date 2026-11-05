"""Маршруты API для расчёта углеродного баланса OIV и экологического аудита (sprint-31)."""

from fastapi import APIRouter, status

from app.api.deps import CurrentUserDep
from app.schemas.errors import ErrorResponse
from app.schemas.sustainability import (
    CarbonBalanceRequest,
    CarbonBalanceResponse,
    OrganicAuditRequest,
    OrganicAuditResponse,
    RegionalCarbonBenchmarkDTO,
)
from app.services.sustainability import (
    audit_organic_practices,
    compute_carbon_balance,
    get_regional_benchmarks,
)

router = APIRouter(
    prefix="/sustainability",
    tags=["sustainability"],
    responses={status.HTTP_401_UNAUTHORIZED: {"model": ErrorResponse}},
)


@router.post(
    "/carbon-balance",
    summary="Расчёт углеродного следа и баланса терруара по стандарту OIV-CST 503-2016",
    response_model=CarbonBalanceResponse,
)
async def get_carbon_balance(
    body: CarbonBalanceRequest,
    _: CurrentUserDep,
) -> CarbonBalanceResponse:
    """Оценивает поглощение CO2 древесиной и почвой против эмиссий тракторов и удобрений."""
    return compute_carbon_balance(body)


@router.post(
    "/organic-audit",
    summary="Аудит соответствия регламентам органического виноделия (EU 2018/848)",
    response_model=OrganicAuditResponse,
)
async def run_organic_audit(
    body: OrganicAuditRequest,
    _: CurrentUserDep,
) -> OrganicAuditResponse:
    """Проверяет соблюдение лимитов меди (<= 4.0 кг/га), запрета гербицидов и залужения междурядий."""
    return audit_organic_practices(body)


@router.get(
    "/benchmarks",
    summary="Региональные углеродные бенчмарки виноградарства Молдовы и Европы",
    response_model=list[RegionalCarbonBenchmarkDTO],
)
async def get_benchmarks(
    _: CurrentUserDep,
) -> list[RegionalCarbonBenchmarkDTO]:
    """Возвращает эталонные показатели углеродоёмкости для сравнения эффективности."""
    return get_regional_benchmarks()
