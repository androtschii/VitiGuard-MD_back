from typing import Literal

from app.schemas.base import ResponseSchema

CheckStatus = Literal["ok", "error"]


class Liveness(ResponseSchema):
    status: Literal["ok"] = "ok"


class Readiness(ResponseSchema):
    status: Literal["ready", "not_ready"]
    checks: dict[str, CheckStatus]
