import uuid
from dataclasses import asdict

from fastapi import APIRouter, Request, status
from fastapi.concurrency import run_in_threadpool

from app.api.deps import (
    CacheDep,
    CurrentUserDep,
    NotifierDep,
    OpenMeteoDep,
    SessionDep,
    SettingsDep,
)
from app.core.errors import ServiceUnavailableError
from app.ml.risk_model import RiskModel
from app.repositories.vineyard import VineyardRepository
from app.schemas.errors import ErrorResponse
from app.schemas.risk import (
    NextDayRiskRead,
    RiskOutlookRead,
    RiskRead,
    TodayRiskRead,
)
from app.services.outlook import vineyard_outlook
from app.services.risk import (
    local_today,
    refresh_risks,
    season_risk_history,
    season_start,
    today_levels,
)
from app.services.risk_forecast import next_day_infection

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
    notifier: NotifierDep,
) -> list[RiskRead]:
    view = await VineyardRepository(session).get_for_owner(user.id, vineyard_id)
    await refresh_risks(session, open_meteo, view.vineyard)
    # Пересчёт мог добавить предупреждения — значок в шапке обновится сразу
    await notifier.alerts_changed(user.id)
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


def get_risk_model(request: Request, settings: SettingsDep) -> RiskModel:
    """Модель загружается при первом запросе и остаётся в памяти процесса."""
    model: RiskModel | None = getattr(request.app.state, "risk_model", None)
    if model is None:
        if not settings.risk_model_path.exists():
            raise ServiceUnavailableError(
                "Модель прогноза риска не загружена на сервер"
            )
        model = RiskModel(settings.risk_model_path)
        request.app.state.risk_model = model
    return model


@router.get(
    "/{vineyard_id}/risks/next-day",
    summary="Заражение милдью на следующий день по модели LightGBM",
    responses={status.HTTP_503_SERVICE_UNAVAILABLE: {"model": ErrorResponse}},
)
async def read_next_day_risk(
    vineyard_id: uuid.UUID,
    request: Request,
    settings: SettingsDep,
    user: CurrentUserDep,
    session: SessionDep,
) -> NextDayRiskRead | None:
    view = await VineyardRepository(session).get_for_owner(user.id, vineyard_id)
    model = await run_in_threadpool(get_risk_model, request, settings)
    # Две недели истории признаков: берётся запас до начала сезона
    result = await next_day_infection(
        session, view.vineyard.id, model, season_start(local_today())
    )
    return NextDayRiskRead.model_validate(asdict(result)) if result else None


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
