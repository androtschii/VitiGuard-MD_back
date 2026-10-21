from datetime import date
from typing import Annotated

from fastapi import APIRouter, Path, Query, Response, status
from fastapi.concurrency import run_in_threadpool

from app.api.deps import CurrentUserDep, SessionDep
from app.schemas.errors import ErrorResponse
from app.services.ndvi_tiles import (
    MAX_ZOOM,
    MIN_ZOOM,
    latest_maps,
    render_tile,
    tile_bounds,
)

router = APIRouter(
    prefix="/tiles",
    tags=["tiles"],
    responses={status.HTTP_401_UNAUTHORIZED: {"model": ErrorResponse}},
)

TILE_HEADERS = {"Cache-Control": "private, max-age=3600"}


# Пустой тайл — 204: MapLibre рисует на его месте прозрачность и не считает
# это ошибкой, а лишних байтов по сети не идёт
@router.get(
    "/ndvi/{z}/{x}/{y}.png",
    summary="Тайл картограмм NDVI участков пользователя",
    response_class=Response,
    responses={
        status.HTTP_200_OK: {"content": {"image/png": {}}},
        status.HTTP_204_NO_CONTENT: {"description": "В тайле нет картограмм"},
    },
)
async def ndvi_tile(
    z: Annotated[int, Path(ge=0, le=MAX_ZOOM)],
    x: Annotated[int, Path(ge=0)],
    y: Annotated[int, Path(ge=0)],
    user: CurrentUserDep,
    session: SessionDep,
    day: Annotated[
        date | None, Query(description="Последний снимок не позже этого дня")
    ] = None,
) -> Response:
    if z < MIN_ZOOM or x >= 2**z or y >= 2**z:
        return Response(status_code=status.HTTP_204_NO_CONTENT)
    maps = await latest_maps(session, user.id, tile_bounds(z, x, y), day)
    png = await run_in_threadpool(render_tile, z, x, y, maps) if maps else None
    if png is None:
        return Response(status_code=status.HTTP_204_NO_CONTENT, headers=TILE_HEADERS)
    return Response(png, media_type="image/png", headers=TILE_HEADERS)
