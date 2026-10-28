from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from prometheus_client.parser import text_string_to_metric_families
from pydantic import SecretStr

from app.core.config import Settings
from app.main import create_app


def samples(text: str, name: str) -> dict[tuple[tuple[str, str], ...], float]:
    """Значения метрики по набору меток."""
    return {
        tuple(sorted(sample.labels.items())): sample.value
        for family in text_string_to_metric_families(text)
        for sample in family.samples
        if sample.name == name
    }


@pytest.fixture
def client() -> Iterator[TestClient]:
    app = create_app(Settings(_env_file=None, environment="test"))

    @app.get("/api/v1/things/{thing_id}")
    async def thing(thing_id: int) -> int:
        return thing_id

    with TestClient(app) as test_client:
        yield test_client


def test_requests_are_counted_by_route_template(client: TestClient) -> None:
    client.get("/api/v1/things/1")
    client.get("/api/v1/things/2")
    client.get("/api/v1/info")

    text = client.get("/metrics").text
    counts = samples(text, "http_requests_total")

    # Два разных id — один ряд по шаблону маршрута
    assert (
        counts[
            (
                ("method", "GET"),
                ("route", "/api/v1/things/{thing_id}"),
                ("status", "200"),
            )
        ]
        == 2
    )
    assert (
        counts[(("method", "GET"), ("route", "/api/v1/info"), ("status", "200"))] == 1
    )
    durations = samples(text, "http_request_duration_seconds_count")
    assert durations[(("method", "GET"), ("route", "/api/v1/things/{thing_id}"))] == 2


def test_unknown_paths_share_one_series(client: TestClient) -> None:
    client.get("/random/a")
    client.get("/random/b")

    counts = samples(client.get("/metrics").text, "http_requests_total")

    assert counts[(("method", "GET"), ("route", "unmatched"), ("status", "404"))] == 2


def test_service_routes_are_not_counted(client: TestClient) -> None:
    client.get("/healthz")
    client.get("/metrics")

    counts = samples(client.get("/metrics").text, "http_requests_total")

    assert not any(
        dict(labels)["route"] in {"/healthz", "/metrics"} for labels in counts
    )


def test_metrics_endpoint_format_and_schema(client: TestClient) -> None:
    response = client.get("/metrics")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert "/metrics" not in client.get("/openapi.json").json()["paths"]


def test_metrics_token_is_required_when_set() -> None:
    settings = Settings(
        _env_file=None, environment="test", metrics_token=SecretStr("s3cret-token")
    )
    with TestClient(create_app(settings)) as client:
        assert client.get("/metrics").status_code == 401
        wrong = client.get("/metrics", headers={"Authorization": "Bearer nope"})
        assert wrong.status_code == 401
        ok = client.get("/metrics", headers={"Authorization": "Bearer s3cret-token"})

    assert ok.status_code == 200
    assert "http_requests_total" in ok.text
