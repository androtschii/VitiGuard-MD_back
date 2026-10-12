import uuid
from typing import Any

from fastapi import APIRouter, status

from app.api.deps import CurrentUserDep, PageParamsDep, SessionDep
from app.models.vineyard import Vineyard
from app.repositories.vineyard import VineyardRepository
from app.schemas.errors import ErrorResponse
from app.schemas.pagination import Page
from app.schemas.vineyard import VineyardCreate, VineyardRead

router = APIRouter(
    prefix="/vineyards",
    tags=["vineyards"],
    responses={status.HTTP_401_UNAUTHORIZED: {"model": ErrorResponse}},
)


def to_read(vineyard: Vineyard, geometry: dict[str, Any]) -> VineyardRead:
    # Контур хранится как геометрия PostGIS, а в ответ идёт GeoJSON из запроса
    return VineyardRead.model_validate(
        {
            "id": vineyard.id,
            "created_at": vineyard.created_at,
            "updated_at": vineyard.updated_at,
            "name": vineyard.name,
            "geometry": geometry,
        }
    )


@router.get("", summary="Мои участки")
async def list_vineyards(
    params: PageParamsDep, user: CurrentUserDep, session: SessionDep
) -> Page[VineyardRead]:
    items, total = await VineyardRepository(session).list_for_owner(user.id, params)
    return Page.of([to_read(*item) for item in items], total, params)


@router.get(
    "/{vineyard_id}",
    summary="Участок",
    responses={status.HTTP_404_NOT_FOUND: {"model": ErrorResponse}},
)
async def read_vineyard(
    vineyard_id: uuid.UUID, user: CurrentUserDep, session: SessionDep
) -> VineyardRead:
    return to_read(
        *await VineyardRepository(session).get_for_owner(user.id, vineyard_id)
    )


@router.post(
    "", status_code=status.HTTP_201_CREATED, summary="Создание участка по контуру"
)
async def create_vineyard(
    data: VineyardCreate, user: CurrentUserDep, session: SessionDep
) -> VineyardRead:
    repository = VineyardRepository(session)
    # Владелец — всегда текущий пользователь: поля owner_id в запросе нет
    vineyard = await repository.create_from_geojson(
        user.id, data.name, data.geometry.model_dump()
    )
    geometry = await repository.geojson(vineyard)
    await session.commit()
    await session.refresh(vineyard)
    return to_read(vineyard, geometry)
