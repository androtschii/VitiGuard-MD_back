from typing import ClassVar

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.schemas.errors import ErrorResponse


class AppError(Exception):
    """Ошибка предметной области. Слои ниже API не знают про HTTP: они бросают
    такие исключения, а в ответ их превращают обработчики ниже."""

    status_code = 500
    headers: ClassVar[dict[str, str] | None] = None

    def __init__(self, detail: str) -> None:
        super().__init__(detail)
        self.detail = detail


class NotFoundError(AppError):
    status_code = 404


class ConflictError(AppError):
    status_code = 409


class UnauthorizedError(AppError):
    """Нет действительных учётных данных: токен отсутствует, подделан или просрочен."""

    status_code = 401
    # По стандарту ответ 401 сообщает, какую схему аутентификации ждёт сервер
    headers: ClassVar[dict[str, str] | None] = {"WWW-Authenticate": "Bearer"}


async def handle_app_error(_: Request, error: Exception) -> JSONResponse:
    assert isinstance(error, AppError)
    body = ErrorResponse(detail=error.detail)
    return JSONResponse(
        status_code=error.status_code,
        content=body.model_dump(),
        headers=error.headers,
    )


async def handle_validation_error(_: Request, error: Exception) -> JSONResponse:
    """Ответ 422 без введённых значений. Стандартный ответ FastAPI возвращает поле
    input с тем, что прислал клиент: для неверного пароля это сам пароль, и он
    попал бы в тело ответа, прокси и логи."""
    assert isinstance(error, RequestValidationError)
    errors = [
        {"loc": item["loc"], "msg": item["msg"], "type": item["type"]}
        for item in error.errors()
    ]
    return JSONResponse(status_code=422, content={"detail": errors})


def register_error_handlers(application: FastAPI) -> None:
    application.add_exception_handler(AppError, handle_app_error)
    application.add_exception_handler(RequestValidationError, handle_validation_error)
