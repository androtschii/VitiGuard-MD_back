import uuid

from fastapi import APIRouter, status
from sqlalchemy import select

from app.api.deps import CurrentUserDep, SessionDep
from app.core.errors import BadRequestError, NotFoundError
from app.models.weather_event import WeatherEvent
from app.repositories.vineyard import VineyardRepository
from app.schemas.errors import ErrorResponse
from app.schemas.weather_event import WeatherEventCreate, WeatherEventRead
from app.services.risk import local_today

router = APIRouter(
    prefix="/vineyards",
    tags=["weather-events"],
    responses={
        status.HTTP_401_UNAUTHORIZED: {"model": ErrorResponse},
        status.HTTP_404_NOT_FOUND: {"model": ErrorResponse},
    },
)


@router.get("/{vineyard_id}/weather-events", summary="Журнал погодных событий")
async def list_events(
    vineyard_id: uuid.UUID, user: CurrentUserDep, session: SessionDep
) -> list[WeatherEventRead]:
    await VineyardRepository(session).get_for_owner(user.id, vineyard_id)
    rows = await session.scalars(
        select(WeatherEvent)
        .where(WeatherEvent.vineyard_id == vineyard_id)
        .order_by(WeatherEvent.occurred_on.desc(), WeatherEvent.created_at.desc())
    )
    return [WeatherEventRead.model_validate(row) for row in rows]


@router.post(
    "/{vineyard_id}/weather-events",
    status_code=status.HTTP_201_CREATED,
    summary="Отметить погодное событие",
    responses={status.HTTP_400_BAD_REQUEST: {"model": ErrorResponse}},
)
async def create_event(
    vineyard_id: uuid.UUID,
    body: WeatherEventCreate,
    user: CurrentUserDep,
    session: SessionDep,
) -> WeatherEventRead:
    await VineyardRepository(session).get_for_owner(user.id, vineyard_id)
    if body.occurred_on > local_today():
        raise BadRequestError("Этот день ещё не наступил")
    event = WeatherEvent(vineyard_id=vineyard_id, **body.model_dump())
    session.add(event)
    await session.commit()
    await session.refresh(event)
    return WeatherEventRead.model_validate(event)


@router.delete(
    "/{vineyard_id}/weather-events/{event_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Удалить погодное событие",
)
async def delete_event(
    vineyard_id: uuid.UUID,
    event_id: uuid.UUID,
    user: CurrentUserDep,
    session: SessionDep,
) -> None:
    await VineyardRepository(session).get_for_owner(user.id, vineyard_id)
    event = await session.get(WeatherEvent, event_id)
    if event is None or event.vineyard_id != vineyard_id:
        raise NotFoundError("Событие не найдено")
    await session.delete(event)
    await session.commit()
