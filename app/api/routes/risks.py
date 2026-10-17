import uuid

from fastapi import APIRouter, status

from app.api.deps import CurrentUserDep, OpenMeteoDep, SessionDep
from app.repositories.vineyard import VineyardRepository
from app.schemas.errors import ErrorResponse
from app.schemas.risk import RiskRead
from app.services.risk import refresh_risks, season_risk_history

router = APIRouter(
    prefix="/vineyards",
    tags=["risks"],
    responses={
        status.HTTP_401_UNAUTHORIZED: {"model": ErrorResponse},
        status.HTTP_404_NOT_FOUND: {"model": ErrorResponse},
    },
)


@router.get("/{vineyard_id}/risks", summary="Риск милдью и оидиума за сезон")
async def read_risks(
    vineyard_id: uuid.UUID, user: CurrentUserDep, session: SessionDep
) -> list[RiskRead]:
    view = await VineyardRepository(session).get_for_owner(user.id, vineyard_id)
    rows = await season_risk_history(session, view.vineyard)
    return [RiskRead.model_validate(row) for row in rows]


# Пересчёт идёт в запросе: два запроса к Open-Meteo и расчёт моделей занимают
# секунды, а пользователь ждёт ответ на экране
@router.post(
    "/{vineyard_id}/risks/refresh",
    summary="Пересчёт риска по свежей погоде и прогнозу на 7 дней",
    responses={status.HTTP_502_BAD_GATEWAY: {"model": ErrorResponse}},
)
async def refresh_vineyard_risks(
    vineyard_id: uuid.UUID,
    user: CurrentUserDep,
    session: SessionDep,
    open_meteo: OpenMeteoDep,
) -> list[RiskRead]:
    view = await VineyardRepository(session).get_for_owner(user.id, vineyard_id)
    await refresh_risks(session, open_meteo, view.vineyard)
    rows = await season_risk_history(session, view.vineyard)
    return [RiskRead.model_validate(row) for row in rows]
