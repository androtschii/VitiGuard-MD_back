"""Маршруты API для биоклиматических индексов терруара и фенологии (sprint-25)."""

import uuid
from datetime import UTC, date, datetime

from fastapi import APIRouter, status
from sqlalchemy import select

from app.api.deps import CurrentUserDep, SessionDep
from app.models.weather import WeatherData
from app.repositories.vineyard import VineyardRepository
from app.schemas.bioclimatic import (
    CoolNightIndexResponse,
    HarvestWindowForecastRequest,
    HarvestWindowForecastResponse,
    HuglinIndexResponse,
    PhenologyEstimateResponse,
    SelyaninovHtcResponse,
    TerroirBioclimaticProfileResponse,
    VarietyCatalogItemResponse,
    WinklerIndexResponse,
)
from app.schemas.errors import ErrorResponse
from app.services.agromodels import DailyWeather, HourlyWeather, daily_weather
from app.services.bioclimatic import (
    VARIETY_CATALOG,
    compute_cool_night_index,
    compute_huglin_index,
    compute_selyaninov_htc,
    compute_winkler_index,
    estimate_phenology_stage,
    get_variety_profile,
    predict_harvest_window,
)

router = APIRouter(
    tags=["bioclimatic"],
    responses={
        status.HTTP_401_UNAUTHORIZED: {"model": ErrorResponse},
        status.HTTP_404_NOT_FOUND: {"model": ErrorResponse},
    },
)


async def _load_vineyard_daily_weather(
    session: SessionDep, vineyard_id: uuid.UUID
) -> list[DailyWeather]:
    rows = await session.scalars(
        select(WeatherData)
        .where(WeatherData.vineyard_id == vineyard_id)
        .order_by(WeatherData.observed_at)
    )
    hours = [
        HourlyWeather(
            at=row.observed_at,
            temperature_c=row.temperature_c,
            relative_humidity=row.relative_humidity,
            precipitation_mm=row.precipitation_mm,
        )
        for row in rows
    ]
    return daily_weather(hours)


@router.get(
    "/vineyards/{vineyard_id}/bioclimatic/phenology-estimate",
    summary="Оценка текущей фенологической фазы BBCH по метеоданным",
    response_model=PhenologyEstimateResponse,
)
async def get_phenology_estimate(
    vineyard_id: uuid.UUID,
    user: CurrentUserDep,
    session: SessionDep,
) -> PhenologyEstimateResponse:
    view = await VineyardRepository(session).get_for_owner(user.id, vineyard_id)
    variety_name = (
        view.vineyard.variety.name if view.vineyard.variety else "feteasca_regala"
    )

    days = await _load_vineyard_daily_weather(session, vineyard_id)

    # Сумма активных температур с 1 марта текущего года
    current_year = date.today().year
    gdd_sum = 0.0
    for day in days:
        if day.day >= date(current_year, 3, 1):
            gdd_sum += max(0.0, day.mean_temp - 10.0)

    estimate = estimate_phenology_stage(gdd_sum, variety_name)
    profile = get_variety_profile(variety_name)

    return PhenologyEstimateResponse(
        current_bbch=estimate.current_bbch,
        current_stage_name=estimate.current_stage_name,
        accumulated_gdd=estimate.accumulated_gdd,
        next_bbch=estimate.next_bbch,
        next_stage_name=estimate.next_stage_name,
        gdd_to_next=estimate.gdd_to_next,
        progress_percentage=estimate.progress_percentage,
        is_season_complete=estimate.is_season_complete,
        variety_name=profile.name,
    )


@router.get(
    "/vineyards/{vineyard_id}/bioclimatic/indices",
    summary="Биоклиматические индексы терруара (Винклер, Гуглин, прохладные ночи, ГТК)",
    response_model=TerroirBioclimaticProfileResponse,
)
async def get_terroir_indices(
    vineyard_id: uuid.UUID,
    user: CurrentUserDep,
    session: SessionDep,
) -> TerroirBioclimaticProfileResponse:
    await VineyardRepository(session).get_for_owner(user.id, vineyard_id)
    days = await _load_vineyard_daily_weather(session, vineyard_id)

    winkler = compute_winkler_index(days)
    huglin = compute_huglin_index(days)
    cool_night = compute_cool_night_index(days)
    selyaninov = compute_selyaninov_htc(days)

    return TerroirBioclimaticProfileResponse(
        winkler=WinklerIndexResponse(
            gdd_sum=winkler.gdd_sum,
            region_class=winkler.region_class,
            region_title=winkler.region_title,
            suitable_varieties=winkler.suitable_varieties,
        ),
        huglin=HuglinIndexResponse(
            hi_value=huglin.hi_value,
            thermal_class=huglin.thermal_class,
            thermal_description=huglin.thermal_description,
        ),
        cool_night=CoolNightIndexResponse(
            ci_temperature=cool_night.ci_temperature,
            night_class=cool_night.night_class,
            aroma_color_potential=cool_night.aroma_color_potential,
        ),
        selyaninov=SelyaninovHtcResponse(
            htc_value=selyaninov.htc_value,
            moisture_class=selyaninov.moisture_class,
            irrigation_need=selyaninov.irrigation_need,
        ),
    )


@router.post(
    "/vineyards/{vineyard_id}/bioclimatic/harvest-window",
    summary="Прогноз окна сбора урожая по прогнозу погоды и созреванию сахаров",
    response_model=HarvestWindowForecastResponse,
)
async def forecast_harvest_window(
    vineyard_id: uuid.UUID,
    body: HarvestWindowForecastRequest,
    user: CurrentUserDep,
    session: SessionDep,
) -> HarvestWindowForecastResponse:
    view = await VineyardRepository(session).get_for_owner(user.id, vineyard_id)
    variety_name = (
        view.vineyard.variety.name if view.vineyard.variety else "feteasca_regala"
    )

    days = await _load_vineyard_daily_weather(session, vineyard_id)
    now = datetime.now(UTC).date()

    history_days = [d for d in days if d.day <= now]
    forecast_days = [d for d in days if d.day > now]

    gdd_sum = sum(
        max(0.0, d.mean_temp - 10.0)
        for d in history_days
        if d.day >= date(now.year, 3, 1)
    )

    pred = predict_harvest_window(
        accumulated_gdd=gdd_sum,
        forecast_days=forecast_days,
        variety=variety_name,
        current_sugar_brix=body.current_sugar_brix,
        target_sugar_brix=body.target_sugar_brix,
    )

    return HarvestWindowForecastResponse(
        variety_name=pred.variety_name,
        target_sugar_brix=pred.target_sugar_brix,
        predicted_brix_trajectory=pred.predicted_brix_trajectory,
        optimal_window_start=pred.optimal_window_start,
        optimal_window_end=pred.optimal_window_end,
        harvest_urgency=pred.harvest_urgency,
        risks_detected=pred.risks_detected,
        agronomic_recommendation=pred.agronomic_recommendation,
    )


@router.get(
    "/bioclimatic/varieties",
    summary="Справочник сортов винограда с термическими фазами BBCH",
    response_model=list[VarietyCatalogItemResponse],
)
async def list_variety_profiles() -> list[VarietyCatalogItemResponse]:
    return [
        VarietyCatalogItemResponse(
            name=profile.name,
            code=profile.code,
            berry_color=profile.berry_color,
            target_brix_default=profile.target_brix_default,
            description=profile.description,
            gdd_thresholds={
                int(stage): gdd for stage, gdd in profile.gdd_thresholds.items()
            },
        )
        for profile in VARIETY_CATALOG.values()
    ]
