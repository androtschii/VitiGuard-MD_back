from typing import Annotated

from fastapi import APIRouter, Query, status

from app.api.deps import CurrentUserDep, OpenMeteoDep
from app.integrations.open_meteo import WeatherSample
from app.schemas.errors import ErrorResponse

router = APIRouter(
    prefix="/weather",
    tags=["weather"],
    responses={
        status.HTTP_401_UNAUTHORIZED: {"model": ErrorResponse},
        status.HTTP_502_BAD_GATEWAY: {"model": ErrorResponse},
    },
)

Latitude = Annotated[float, Query(ge=-90, le=90, description="Широта, градусы")]
Longitude = Annotated[float, Query(ge=-180, le=180, description="Долгота, градусы")]


# Только для вошедших: иначе эндпоинт стал бы бесплатным посредником к Open-Meteo
# для кого угодно, а лимит запросов к сервису общий для всего сервера
@router.get("/current", summary="Текущая погода в точке")
async def current_weather(
    latitude: Latitude,
    longitude: Longitude,
    _: CurrentUserDep,
    open_meteo: OpenMeteoDep,
) -> WeatherSample:
    return await open_meteo.current(latitude, longitude)
