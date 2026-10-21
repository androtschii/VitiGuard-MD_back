import uuid
from dataclasses import asdict

from fastapi import APIRouter, status

from app.api.deps import CacheDep, CurrentUserDep, OpenMeteoDep, SessionDep
from app.repositories.vineyard import VineyardRepository
from app.schemas.errors import ErrorResponse
from app.schemas.risk import RiskOutlookRead, RiskRead, TodayRiskRead
from app.services.outlook import vineyard_outlook
from app.services.risk import (
    local_today,
    refresh_risks,
    season_risk_history,
    today_levels,
)

# Ансамбль Open-Meteo обновляется несколько раз в сутки, а его загрузка и
# расчёт моделей по 40 членам — самый тяжёлый запрос риска: вероятность хранится час
OUTLOOK_TTL_SECONDS = 3600

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


@router.get(
    "/{vineyard_id}/risks/outlook",
    summary="Вероятность вспышки милдью и оидиума на 7 дней",
    responses={status.HTTP_502_BAD_GATEWAY: {"model": ErrorResponse}},
)
async def read_risk_outlook(
    vineyard_id: uuid.UUID,
    user: CurrentUserDep,
    session: SessionDep,
    open_meteo: OpenMeteoDep,
    cache: CacheDep,
) -> RiskOutlookRead:
    view = await VineyardRepository(session).get_for_owner(user.id, vineyard_id)
    key = f"risks:outlook:{vineyard_id}:{local_today()}"
    stored = await cache.get_json(key)
    if stored is not None:
        return RiskOutlookRead.model_validate(stored)
    outlook = RiskOutlookRead.model_validate(
        asdict(await vineyard_outlook(session, open_meteo, view.vineyard))
    )
    await cache.set_json(key, outlook.model_dump(mode="json"), OUTLOOK_TTL_SECONDS)
    return outlook


# Отдельный префикс: путь /vineyards/... конфликтовал бы с /vineyards/{id}
today_router = APIRouter(
    prefix="/risks",
    tags=["risks"],
    responses={status.HTTP_401_UNAUTHORIZED: {"model": ErrorResponse}},
)


@today_router.get("/today", summary="Уровни риска на сегодня по моим участкам")
async def read_today_risks(
    user: CurrentUserDep, session: SessionDep
) -> list[TodayRiskRead]:
    return [
        TodayRiskRead.model_validate(vars(levels))
        for levels in await today_levels(session, user.id)
    ]
