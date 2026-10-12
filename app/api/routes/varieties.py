import uuid
from typing import Any

from fastapi import APIRouter, status

from app.api.deps import AgronomistDep, CurrentUserDep, SessionDep
from app.repositories.variety import VarietyRepository
from app.schemas.errors import ErrorResponse
from app.schemas.variety import VarietyCreate, VarietyRead, VarietyUpdate

# Справочник читают все, а меняют агроном и администратор: ошибка в
# восприимчивости сорта искажает риски у всех хозяйств с этим сортом
router = APIRouter(
    prefix="/varieties",
    tags=["varieties"],
    responses={status.HTTP_401_UNAUTHORIZED: {"model": ErrorResponse}},
)

Responses = dict[int | str, dict[str, Any]]

NOT_FOUND: Responses = {status.HTTP_404_NOT_FOUND: {"model": ErrorResponse}}
EDIT_ERRORS: Responses = {
    status.HTTP_403_FORBIDDEN: {"model": ErrorResponse},
    status.HTTP_409_CONFLICT: {"model": ErrorResponse},
}


@router.get("", summary="Сорта винограда")
async def list_varieties(_: CurrentUserDep, session: SessionDep) -> list[VarietyRead]:
    varieties = await VarietyRepository(session).all()
    return [VarietyRead.model_validate(variety) for variety in varieties]


@router.get("/{variety_id}", summary="Сорт", responses=NOT_FOUND)
async def read_variety(
    variety_id: uuid.UUID, _: CurrentUserDep, session: SessionDep
) -> VarietyRead:
    variety = await VarietyRepository(session).get_or_raise(variety_id)
    return VarietyRead.model_validate(variety)


@router.post(
    "",
    status_code=status.HTTP_201_CREATED,
    summary="Новый сорт",
    responses=EDIT_ERRORS,
)
async def create_variety(
    data: VarietyCreate, _: AgronomistDep, session: SessionDep
) -> VarietyRead:
    variety = await VarietyRepository(session).create(**data.model_dump())
    await session.commit()
    await session.refresh(variety)
    return VarietyRead.model_validate(variety)


@router.patch(
    "/{variety_id}",
    summary="Изменение сорта",
    responses={**NOT_FOUND, **EDIT_ERRORS},
)
async def update_variety(
    variety_id: uuid.UUID, data: VarietyUpdate, _: AgronomistDep, session: SessionDep
) -> VarietyRead:
    repository = VarietyRepository(session)
    variety = await repository.get_or_raise(variety_id)
    updated = await repository.update(variety, **data.model_dump(exclude_unset=True))
    await session.commit()
    return VarietyRead.model_validate(updated)


@router.delete(
    "/{variety_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Удаление сорта",
    responses={**NOT_FOUND, status.HTTP_403_FORBIDDEN: {"model": ErrorResponse}},
)
async def delete_variety(
    variety_id: uuid.UUID, _: AgronomistDep, session: SessionDep
) -> None:
    repository = VarietyRepository(session)
    await repository.delete(await repository.get_or_raise(variety_id))
    await session.commit()
