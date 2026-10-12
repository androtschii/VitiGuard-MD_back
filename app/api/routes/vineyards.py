from fastapi import APIRouter, status

from app.api.deps import CurrentUserDep, SessionDep
from app.repositories.vineyard import VineyardRepository
from app.schemas.errors import ErrorResponse
from app.schemas.vineyard import VineyardCreate, VineyardRead

router = APIRouter(
    prefix="/vineyards",
    tags=["vineyards"],
    responses={status.HTTP_401_UNAUTHORIZED: {"model": ErrorResponse}},
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
    return VineyardRead.model_validate(
        {
            "id": vineyard.id,
            "created_at": vineyard.created_at,
            "updated_at": vineyard.updated_at,
            "name": vineyard.name,
            "geometry": geometry,
        }
    )
