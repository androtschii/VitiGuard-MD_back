import uuid
from datetime import date

from app.schemas.base import EntityResponse


class FieldNoteRead(EntityResponse):
    """Заметка с обхода участка."""

    vineyard_id: uuid.UUID
    noted_on: date
    text: str
    latitude: float | None
    longitude: float | None
    has_photo: bool
