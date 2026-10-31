import uuid

from fastapi import APIRouter, status
from sqlalchemy import select

from app.api.deps import CurrentUserDep, SessionDep
from app.core.errors import BadRequestError, NotFoundError
from app.models.phenology import PhenologyObservation
from app.repositories.vineyard import VineyardRepository
from app.schemas.errors import ErrorResponse
from app.schemas.phenology import PhenologyCreate, PhenologyRead
from app.services.risk import local_today

router = APIRouter(
    prefix="/vineyards",
    tags=["phenology"],
    responses={
        status.HTTP_401_UNAUTHORIZED: {"model": ErrorResponse},
        status.HTTP_404_NOT_FOUND: {"model": ErrorResponse},
    },
)


@router.get("/{vineyard_id}/phenology", summary="Фазы развития лозы на участке")
async def list_phenology(
    vineyard_id: uuid.UUID, user: CurrentUserDep, session: SessionDep
) -> list[PhenologyRead]:
    await VineyardRepository(session).get_for_owner(user.id, vineyard_id)
    rows = await session.scalars(
        select(PhenologyObservation)
        .where(PhenologyObservation.vineyard_id == vineyard_id)
        .order_by(PhenologyObservation.observed_on.desc())
    )
    return [PhenologyRead.model_validate(row) for row in rows]


@router.post(
    "/{vineyard_id}/phenology",
    status_code=status.HTTP_201_CREATED,
    summary="Отметить фазу: повторная отметка за тот же день заменяет прежнюю",
    responses={status.HTTP_400_BAD_REQUEST: {"model": ErrorResponse}},
)
async def save_phenology(
    vineyard_id: uuid.UUID,
    body: PhenologyCreate,
    user: CurrentUserDep,
    session: SessionDep,
) -> PhenologyRead:
    await VineyardRepository(session).get_for_owner(user.id, vineyard_id)
    if body.observed_on > local_today():
        raise BadRequestError("День наблюдения ещё не наступил")
    row = await session.scalar(
        select(PhenologyObservation).where(
            PhenologyObservation.vineyard_id == vineyard_id,
            PhenologyObservation.observed_on == body.observed_on,
        )
    )
    if row is None:
        row = PhenologyObservation(vineyard_id=vineyard_id, **body.model_dump())
        session.add(row)
    else:
        row.stage = body.stage
    await session.commit()
    await session.refresh(row)
    return PhenologyRead.model_validate(row)


@router.delete(
    "/{vineyard_id}/phenology/{observation_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Удалить отметку фазы",
)
async def delete_phenology(
    vineyard_id: uuid.UUID,
    observation_id: uuid.UUID,
    user: CurrentUserDep,
    session: SessionDep,
) -> None:
    await VineyardRepository(session).get_for_owner(user.id, vineyard_id)
    row = await session.get(PhenologyObservation, observation_id)
    if row is None or row.vineyard_id != vineyard_id:
        raise NotFoundError("Отметка не найдена")
    await session.delete(row)
    await session.commit()
