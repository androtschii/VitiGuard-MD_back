import uuid
from typing import Annotated, Any

from fastapi import APIRouter, Query, Response, status
from fastapi.concurrency import run_in_threadpool
from sqlalchemy import func, select
from sqlalchemy.orm import selectinload

from app.api.deps import CurrentUserDep, SessionDep, StorageDep
from app.core.errors import NotFoundError
from app.models.vine_check import VineCheck
from app.schemas.diagnosis import VineCheckRead, VineDetectionItem
from app.schemas.errors import ErrorResponse
from app.schemas.pagination import Page, PageParams

# История анализов фото ряда: стволы и грозди, как у робота-сборщика
router = APIRouter(
    prefix="/vine-checks",
    tags=["vine-checks"],
    responses={status.HTTP_401_UNAUTHORIZED: {"model": ErrorResponse}},
)
NOT_FOUND: dict[int | str, dict[str, Any]] = {
    status.HTTP_404_NOT_FOUND: {"model": ErrorResponse}
}


def to_read(check: VineCheck) -> VineCheckRead:
    return VineCheckRead(
        id=check.id,
        created_at=check.created_at,
        updated_at=check.updated_at,
        vineyard_id=check.vineyard_id,
        vineyard_name=check.vineyard.name if check.vineyard else None,
        model=check.model_name,
        image_width=check.width,
        image_height=check.height,
        trunks=check.trunks,
        bunches=check.bunches,
        detections=[VineDetectionItem(**item) for item in check.detections],
    )


async def owned(
    session: SessionDep, user_id: uuid.UUID, check_id: uuid.UUID
) -> VineCheck:
    """Чужой анализ неотличим от несуществующего."""
    check = await session.scalar(
        select(VineCheck)
        .where(VineCheck.id == check_id, VineCheck.owner_id == user_id)
        .options(selectinload(VineCheck.vineyard))
    )
    if check is None:
        raise NotFoundError("Анализ ряда не найден")
    return check


@router.get("", summary="История анализов фото ряда")
async def list_vine_checks(
    params: Annotated[PageParams, Query()],
    user: CurrentUserDep,
    session: SessionDep,
) -> Page[VineCheckRead]:
    query = select(VineCheck).where(VineCheck.owner_id == user.id)
    total = await session.scalar(select(func.count()).select_from(query.subquery()))
    rows = await session.scalars(
        query.options(selectinload(VineCheck.vineyard))
        .order_by(VineCheck.created_at.desc(), VineCheck.id.desc())
        .offset(params.offset)
        .limit(params.size)
    )
    return Page.of([to_read(row) for row in rows], total or 0, params)


@router.get(
    "/{check_id}/image",
    summary="Фото ряда",
    response_class=Response,
    responses={
        **NOT_FOUND,
        status.HTTP_200_OK: {"content": {"image/jpeg": {}, "image/png": {}}},
    },
)
async def vine_check_image(
    check_id: uuid.UUID,
    user: CurrentUserDep,
    session: SessionDep,
    storage: StorageDep,
) -> Response:
    check = await owned(session, user.id, check_id)
    data = await run_in_threadpool(storage.get, check.storage_key)
    return Response(
        data,
        media_type=check.content_type,
        headers={"Cache-Control": "private, max-age=86400"},
    )


@router.delete(
    "/{check_id}",
    summary="Удаление анализа ряда вместе с фото",
    status_code=status.HTTP_204_NO_CONTENT,
    responses=NOT_FOUND,
)
async def delete_vine_check(
    check_id: uuid.UUID,
    user: CurrentUserDep,
    session: SessionDep,
    storage: StorageDep,
) -> None:
    check = await owned(session, user.id, check_id)
    key = check.storage_key
    await session.delete(check)
    await session.commit()
    await run_in_threadpool(storage.delete, key)
