import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.core.errors import AppError, ConflictError, NotFoundError
from app.main import create_app


@pytest.mark.parametrize(
    ("error", "status"),
    [
        (NotFoundError("Участок не найден"), 404),
        (ConflictError("Участок уже есть"), 409),
        (AppError("Что-то сломалось"), 500),
    ],
)
def test_domain_errors_become_json_responses(error: AppError, status: int) -> None:
    app = create_app(Settings(_env_file=None, environment="test"))

    @app.get("/fail")
    async def fail() -> None:
        raise error

    with TestClient(app) as client:
        response = client.get("/fail")

    assert response.status_code == status
    assert response.json() == {"detail": error.detail}
