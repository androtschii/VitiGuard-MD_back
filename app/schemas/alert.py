import uuid
from datetime import date

from pydantic import AwareDatetime, Field

from app.models.enums import Disease, RiskLevel
from app.schemas.base import EntityResponse, ResponseSchema
from app.schemas.pagination import PageParams


class AlertQuery(PageParams):
    """Страница предупреждений: ?unread=true — только непрочитанные."""

    unread: bool = False


class AlertRead(EntityResponse):
    """Предупреждение о риске болезни на участке на день (pr-080)."""

    vineyard_id: uuid.UUID
    vineyard_name: str
    disease: Disease
    # День ожидаемого риска
    day: date
    level: RiskLevel
    # None — не прочитано
    read_at: AwareDatetime | None
    # Риск на день после сегодняшнего — по прогнозу погоды
    forecast: bool


class UnreadCount(ResponseSchema):
    """Число непрочитанных предупреждений."""

    count: int = Field(ge=0)


class ReadAllResult(ResponseSchema):
    """Сколько предупреждений отмечено прочитанными."""

    updated: int = Field(ge=0)
