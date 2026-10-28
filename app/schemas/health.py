from typing import Literal

from app.schemas.base import ResponseSchema

CheckStatus = Literal["ok", "error"]


class Liveness(ResponseSchema):
    """Ответ проверки живости процесса."""

    status: Literal["ok"] = "ok"


class Readiness(ResponseSchema):
    """Готовность сервиса: состояние базы данных и Redis."""

    status: Literal["ready", "not_ready"]
    checks: dict[str, CheckStatus]
