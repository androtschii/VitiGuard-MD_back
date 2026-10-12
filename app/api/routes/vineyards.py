import uuid
from typing import Annotated

from fastapi import APIRouter, Query, status

from app.api.deps import CurrentUserDep, PageParamsDep, SessionDep
from app.repositories.vineyard import VineyardRepository, VineyardView
from app.schemas.errors import ErrorResponse
from app.schemas.pagination import Page
from app.schemas.vineyard import (
    NearbyQuery,
    VineyardCreate,
    VineyardNearby,
    VineyardRead,
)
from app.services.vineyard import validate_geometry

router = APIRouter(
    prefix="/vineyards",
    tags=["vineyards"],
    responses={status.HTTP_401_UNAUTHORIZED: {"model": ErrorResponse}},
)


def to_read(view: VineyardView) -> VineyardRead:
    # Контур хранится как геометрия PostGIS, а в ответ идут GeoJSON и площадь из запроса
    return VineyardRead.model_validate(
        {
            "id": view.vineyard.id,
            "created_at": view.vineyard.created_at,
            "updated_at": view.vineyard.updated_at,
            "name": view.vineyard.name,
            "geometry": view.geometry,
            "area_ha": view.area_ha,
        }
    )


@router.get("", summary="Мои участки")
async def list_vineyards(
    params: PageParamsDep, user: CurrentUserDep, session: SessionDep
) -> Page[VineyardRead]:
    items, total = await VineyardRepository(session).list_for_owner(user.id, params)
    return Page.of([to_read(item) for item in items], total, params)


# Объявлен раньше /{vineyard_id}: иначе «nearby» разбиралось бы как id участка
@router.get("/nearby", summary="Ближайшие участки к точке")
async def nearby_vineyards(
    query: Annotated[NearbyQuery, Query()], user: CurrentUserDep, session: SessionDep
) -> list[VineyardNearby]:
    found = await VineyardRepository(session).nearest(
        user.id, query.lon, query.lat, query.limit, query.max_distance_km * 1000
    )
    return [
        VineyardNearby.model_validate(
            {**to_read(view).model_dump(), "distance_m": distance}
        )
        for view, distance in found
    ]


@router.get(
    "/{vineyard_id}",
    summary="Участок",
    responses={status.HTTP_404_NOT_FOUND: {"model": ErrorResponse}},
)
async def read_vineyard(
    vineyard_id: uuid.UUID, user: CurrentUserDep, session: SessionDep
) -> VineyardRead:
    return to_read(
        await VineyardRepository(session).get_for_owner(user.id, vineyard_id)
    )


@router.post(
    "",
    status_code=status.HTTP_201_CREATED,
    summary="Создание участка по контуру",
    responses={
        status.HTTP_400_BAD_REQUEST: {"model": ErrorResponse},
        status.HTTP_409_CONFLICT: {"model": ErrorResponse},
    },
)
async def create_vineyard(
    data: VineyardCreate, user: CurrentUserDep, session: SessionDep
) -> VineyardRead:
    geometry = data.geometry.model_dump()
    await validate_geometry(session, user.id, geometry)
    repository = VineyardRepository(session)
    # Владелец — всегда текущий пользователь: поля owner_id в запросе нет
    vineyard = await repository.create_from_geojson(user.id, data.name, geometry)
    await session.commit()
    await session.refresh(vineyard)
    return to_read(await repository.view(vineyard))
