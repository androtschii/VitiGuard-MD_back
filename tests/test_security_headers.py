from typing import Any

from fastapi.responses import PlainTextResponse
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app

CLIENT_ORIGIN = "https://vitiguard.md"
PREFLIGHT = {
    "Access-Control-Request-Method": "POST",
    "Access-Control-Request-Headers": "Authorization, Content-Type",
}


def make_client(**overrides: Any) -> TestClient:
    settings = Settings(_env_file=None, **{"environment": "test", **overrides})
    return TestClient(create_app(settings))


def test_security_headers_on_regular_and_error_responses() -> None:
    with make_client() as client:
        for response in (client.get("/healthz"), client.get("/nowhere")):
            assert response.headers["X-Content-Type-Options"] == "nosniff"
            assert response.headers["X-Frame-Options"] == "DENY"
            assert (
                response.headers["Referrer-Policy"] == "strict-origin-when-cross-origin"
            )
            assert "microphone=()" in response.headers["Permissions-Policy"]
            # Не production: HSTS на машине разработчика сломал бы http://127.0.0.1
            assert "Strict-Transport-Security" not in response.headers


def test_hsts_only_in_production() -> None:
    with make_client(environment="production", secret_key="s" * 40) as client:
        response = client.get("/healthz")

    assert response.headers["Strict-Transport-Security"] == (
        "max-age=31536000; includeSubDomains"
    )


def test_route_header_is_not_overwritten() -> None:
    app = create_app(Settings(_env_file=None, environment="test"))

    @app.get("/framed")
    async def framed() -> PlainTextResponse:
        return PlainTextResponse("ok", headers={"X-Frame-Options": "SAMEORIGIN"})

    with TestClient(app) as client:
        assert client.get("/framed").headers["X-Frame-Options"] == "SAMEORIGIN"


def test_cors_allows_configured_origin_with_credentials() -> None:
    with make_client(cors_origins=[CLIENT_ORIGIN]) as client:
        preflight = client.options(
            "/api/v1/info", headers={"Origin": CLIENT_ORIGIN, **PREFLIGHT}
        )
        response = client.get("/api/v1/info", headers={"Origin": CLIENT_ORIGIN})

    assert preflight.status_code == 200
    assert preflight.headers["Access-Control-Allow-Origin"] == CLIENT_ORIGIN
    assert preflight.headers["Access-Control-Allow-Credentials"] == "true"
    # Заголовки безопасности — и на preflight-ответе
    assert preflight.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["Access-Control-Allow-Origin"] == CLIENT_ORIGIN
    assert "Content-Disposition" in response.headers["Access-Control-Expose-Headers"]


def test_cors_rejects_other_origin() -> None:
    with make_client(cors_origins=[CLIENT_ORIGIN]) as client:
        preflight = client.options(
            "/api/v1/info", headers={"Origin": "https://evil.example", **PREFLIGHT}
        )

    assert preflight.status_code == 400
    assert "Access-Control-Allow-Origin" not in preflight.headers


def test_cors_is_off_without_origins() -> None:
    with make_client() as client:
        response = client.get("/api/v1/info", headers={"Origin": CLIENT_ORIGIN})

    assert response.status_code == 200
    assert "Access-Control-Allow-Origin" not in response.headers
