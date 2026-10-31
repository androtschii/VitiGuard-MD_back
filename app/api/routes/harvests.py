import uuid

from fastapi import APIRouter, status
from sqlalchemy import select

from app.api.deps import CurrentUserDep, SessionDep
from app.core.errors import BadRequestError, NotFoundError
from app.models.harvest import HarvestRecord
from app.repositories.vineyard import VineyardRepository
from app.schemas.errors import ErrorResponse
from app.schemas.harvest import HarvestRead, HarvestSave
from app.services.risk import local_today

router = APIRouter(
    prefix="/vineyards",
    tags=["harvests"],
    responses={
        status.HTTP_401_UNAUTHORIZED: {"model": ErrorResponse},
        status.HTTP_404_NOT_FOUND: {"model": ErrorResponse},
    },
)


@router.get("/{vineyard_id}/harvests", summary="Урожай участка по сезонам")
async def list_harvests(
    vineyard_id: uuid.UUID, user: CurrentUserDep, session: SessionDep
) -> list[HarvestRead]:
    await VineyardRepository(session).get_for_owner(user.id, vineyard_id)
    rows = await session.scalars(
        select(HarvestRecord)
        .where(HarvestRecord.vineyard_id == vineyard_id)
        .order_by(HarvestRecord.season.desc())
    )
    return [HarvestRead.model_validate(row) for row in rows]


@router.post(
    "/{vineyard_id}/harvests",
    status_code=status.HTTP_201_CREATED,
    summary="Сохранить урожай: запись того же сезона заменяется",
    responses={status.HTTP_400_BAD_REQUEST: {"model": ErrorResponse}},
)
async def save_harvest(
    vineyard_id: uuid.UUID,
    body: HarvestSave,
    user: CurrentUserDep,
    session: SessionDep,
) -> HarvestRead:
    await VineyardRepository(session).get_for_owner(user.id, vineyard_id)
    if body.season > local_today().year:
        raise BadRequestError("Этот сезон ещё не наступил")
    if body.estimated_kg is None and body.actual_kg is None:
        raise BadRequestError("Укажите прогноз или фактический урожай")
    row = await session.scalar(
        select(HarvestRecord).where(
            HarvestRecord.vineyard_id == vineyard_id,
            HarvestRecord.season == body.season,
        )
    )
    if row is None:
        row = HarvestRecord(vineyard_id=vineyard_id, **body.model_dump())
        session.add(row)
    else:
        row.estimated_kg = body.estimated_kg
        row.actual_kg = body.actual_kg
        row.sugar_brix = body.sugar_brix
    await session.commit()
    await session.refresh(row)
    return HarvestRead.model_validate(row)


@router.delete(
    "/{vineyard_id}/harvests/{harvest_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Удалить запись об урожае",
)
async def delete_harvest(
    vineyard_id: uuid.UUID,
    harvest_id: uuid.UUID,
    user: CurrentUserDep,
    session: SessionDep,
) -> None:
    await VineyardRepository(session).get_for_owner(user.id, vineyard_id)
    row = await session.get(HarvestRecord, harvest_id)
    if row is None or row.vineyard_id != vineyard_id:
        raise NotFoundError("Запись не найдена")
    await session.delete(row)
    await session.commit()
