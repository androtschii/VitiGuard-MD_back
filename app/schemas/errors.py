from app.schemas.base import ResponseSchema


class ErrorResponse(ResponseSchema):
    """Ошибка в том виде, в каком её отдаёт FastAPI (HTTPException): фронтенд
    показывает поле detail пользователю."""

    detail: str
