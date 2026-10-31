import uuid
from datetime import date, timedelta
from typing import Annotated

from fastapi import APIRouter, File, Form, Response, UploadFile, status
from fastapi.concurrency import run_in_threadpool
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import CurrentUserDep, SessionDep, StorageDep
from app.core.errors import BadRequestError, NotFoundError, PayloadTooLargeError
from app.models.field_note import FieldNote
from app.repositories.vineyard import VineyardRepository
from app.schemas.errors import ErrorResponse
from app.schemas.field_note import FieldNoteRead
from app.services.diagnosis import photo_location, read_leaf_image
from app.services.diagnosis_history import CONTENT_TYPES, EXTENSIONS
from app.services.risk import local_today

router = APIRouter(
    prefix="/vineyards",
    tags=["field-notes"],
    responses={
        status.HTTP_401_UNAUTHORIZED: {"model": ErrorResponse},
        status.HTTP_404_NOT_FOUND: {"model": ErrorResponse},
    },
)

MAX_PHOTO_BYTES = 10 * 1024 * 1024


def to_read(note: FieldNote) -> FieldNoteRead:
    return FieldNoteRead(
        id=note.id,
        created_at=note.created_at,
        updated_at=note.updated_at,
        vineyard_id=note.vineyard_id,
        noted_on=note.noted_on,
        text=note.text,
        latitude=note.latitude,
        longitude=note.longitude,
        has_photo=note.photo_key is not None,
    )


async def get_note(
    session: AsyncSession, vineyard_id: uuid.UUID, note_id: uuid.UUID
) -> FieldNote:
    note = await session.get(FieldNote, note_id)
    if note is None or note.vineyard_id != vineyard_id:
        raise NotFoundError("Заметка не найдена")
    return note


@router.get("/{vineyard_id}/notes", summary="Заметки с обходов участка")
async def list_notes(
    vineyard_id: uuid.UUID, user: CurrentUserDep, session: SessionDep
) -> list[FieldNoteRead]:
    await VineyardRepository(session).get_for_owner(user.id, vineyard_id)
    rows = await session.scalars(
        select(FieldNote)
        .where(FieldNote.vineyard_id == vineyard_id)
        .order_by(FieldNote.noted_on.desc(), FieldNote.created_at.desc())
    )
    return [to_read(row) for row in rows]


@router.post(
    "/{vineyard_id}/notes",
    status_code=status.HTTP_201_CREATED,
    summary="Добавить заметку с обхода",
    responses={
        status.HTTP_400_BAD_REQUEST: {"model": ErrorResponse},
        status.HTTP_413_CONTENT_TOO_LARGE: {"model": ErrorResponse},
        status.HTTP_415_UNSUPPORTED_MEDIA_TYPE: {"model": ErrorResponse},
        status.HTTP_503_SERVICE_UNAVAILABLE: {"model": ErrorResponse},
    },
)
async def create_note(
    vineyard_id: uuid.UUID,
    noted_on: Annotated[date, Form(description="День обхода")],
    user: CurrentUserDep,
    session: SessionDep,
    storage: StorageDep,
    text: Annotated[str, Form(max_length=2000)] = "",
    latitude: Annotated[float | None, Form(ge=-90, le=90)] = None,
    longitude: Annotated[float | None, Form(ge=-180, le=180)] = None,
    client_id: Annotated[
        uuid.UUID | None, Form(description="Метка устройства против повторной записи")
    ] = None,
    photo: Annotated[
        UploadFile | None, File(description="Фото: JPEG, PNG или WebP до 10 МБ")
    ] = None,
) -> FieldNoteRead:
    await VineyardRepository(session).get_for_owner(user.id, vineyard_id)
    # Сутки запаса: заметка могла ждать в очереди устройства в другом часовом поясе
    if noted_on > local_today() + timedelta(days=1):
        raise BadRequestError("День обхода ещё не наступил")
    if (latitude is None) != (longitude is None):
        raise BadRequestError("Нужны обе координаты точки")
    text = text.strip()
    if not text and photo is None:
        raise BadRequestError("Заметке нужен текст или фото")
    if client_id is not None:
        existing = await session.scalar(
            select(FieldNote).where(
                FieldNote.vineyard_id == vineyard_id, FieldNote.client_id == client_id
            )
        )
        if existing is not None:
            return to_read(existing)

    note = FieldNote(
        vineyard_id=vineyard_id,
        noted_on=noted_on,
        text=text,
        latitude=latitude,
        longitude=longitude,
        client_id=client_id,
    )
    session.add(note)
    await session.flush()

    key = None
    if photo is not None:
        data = await photo.read(MAX_PHOTO_BYTES + 1)
        if len(data) > MAX_PHOTO_BYTES:
            raise PayloadTooLargeError("Файл больше 10 МБ")
        image = read_leaf_image(data)
        extension = EXTENSIONS.get(image.format or "", "jpg")
        key = f"{user.id}/notes/{note.id}.{extension}"
        note.photo_key = key
        note.photo_type = CONTENT_TYPES[extension]
        # Точку не указали: берём её из снимка, если телефон её записал
        location = photo_location(data)
        if latitude is None and location is not None:
            note.latitude, note.longitude = location.lat, location.lon
        await run_in_threadpool(storage.put, key, data, note.photo_type)
    try:
        await session.commit()
    except Exception:
        if key is not None:
            await run_in_threadpool(storage.delete, key)
        raise
    await session.refresh(note)
    return to_read(note)


@router.get(
    "/{vineyard_id}/notes/{note_id}/photo",
    summary="Фото заметки",
    response_class=Response,
    responses={status.HTTP_200_OK: {"content": {"image/jpeg": {}, "image/png": {}}}},
)
async def note_photo(
    vineyard_id: uuid.UUID,
    note_id: uuid.UUID,
    user: CurrentUserDep,
    session: SessionDep,
    storage: StorageDep,
) -> Response:
    await VineyardRepository(session).get_for_owner(user.id, vineyard_id)
    note = await get_note(session, vineyard_id, note_id)
    if note.photo_key is None:
        raise NotFoundError("У заметки нет фото")
    data = await run_in_threadpool(storage.get, note.photo_key)
    return Response(
        data,
        media_type=note.photo_type,
        headers={"Cache-Control": "private, max-age=86400"},
    )


@router.delete(
    "/{vineyard_id}/notes/{note_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Удалить заметку вместе с фото",
)
async def delete_note(
    vineyard_id: uuid.UUID,
    note_id: uuid.UUID,
    user: CurrentUserDep,
    session: SessionDep,
    storage: StorageDep,
) -> None:
    await VineyardRepository(session).get_for_owner(user.id, vineyard_id)
    note = await get_note(session, vineyard_id, note_id)
    key = note.photo_key
    await session.delete(note)
    await session.commit()
    if key is not None:
        await run_in_threadpool(storage.delete, key)
