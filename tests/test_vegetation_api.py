from collections.abc import Callable
from datetime import UTC, date, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient
from raster_scene import CLOUD, PARCEL, make_scene

from app.api.routes import vineyards as vineyard_routes
from app.core.config import Settings
from app.integrations.earth_search import Band, EarthScene, EarthSearchClient
from app.services import vegetation_history
from app.tasks import satellite
from app.worker.celery_app import celery_app

pytestmark = pytest.mark.integration

RunSql = Callable[[str], list[tuple[object, ...]]]
CREDENTIALS = {"email": "grower@example.md", "password": "secret123"}


def login(client: TestClient, email: str = CREDENTIALS["email"]) -> dict[str, str]:
    credentials = {**CREDENTIALS, "email": email}
    client.post("/api/v1/auth/register", json={**credentials, "full_name": "Ион"})
    token = client.post("/api/v1/auth/login", json=credentials).json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def create_vineyard(client: TestClient, headers: dict[str, str]) -> str:
    response = client.post(
        "/api/v1/vineyards",
        headers=headers,
        json={"name": "Криково", "geometry": PARCEL},
    )
    vineyard_id: str = response.json()["id"]
    return vineyard_id


def scenes(tmp_path: Path) -> list[EarthScene]:
    """Чистый снимок, снимок с облаками над участком и снимок, каналы которого
    не читаются (сбой хранилища)."""
    clear = make_scene(tmp_path)
    cloudy = make_scene(
        tmp_path,
        scl=CLOUD,
        product_id="S2B_35TPN_20260603_0_L2A",
        acquired_at=datetime(2026, 6, 3, 9, 7, tzinfo=UTC),
    )
    broken = make_scene(
        tmp_path,
        product_id="S2A_35TPN_20260715_1_L2A",
        acquired_at=datetime(2026, 7, 15, 9, 17, tzinfo=UTC),
    )
    broken.bands["B08"] = Band(href=str(tmp_path / "missing.tif"))
    return [clear, cloudy, broken]


@pytest.fixture
def process(
    migrated_database_url: str, monkeypatch: pytest.MonkeyPatch
) -> Callable[[str, list[EarthScene]], dict[str, int]]:
    """Задача Celery без сети: каталог Earth Search подменён готовым списком."""

    def run(vineyard_id: str, found: list[EarthScene]) -> dict[str, int]:
        async def fake_search(*_: Any, **__: Any) -> list[EarthScene]:
            return found

        monkeypatch.setattr(EarthSearchClient, "search", fake_search)
        settings = Settings(
            _env_file=None,
            environment="test",
            database_url=migrated_database_url,  # type: ignore[arg-type]
        )
        monkeypatch.setattr(satellite, "get_settings", lambda: settings)
        return satellite.process_vineyard_task(
            vineyard_id, "2026-03-01T00:00:00+00:00", "2026-10-16T23:59:59+00:00"
        )

    return run


def test_processing_saves_indices_and_history(
    api_client: TestClient,
    tmp_path: Path,
    process: Callable[[str, list[EarthScene]], dict[str, int]],
    run_sql: RunSql,
) -> None:
    headers = login(api_client)
    vineyard_id = create_vineyard(api_client, headers)
    found = scenes(tmp_path)

    first = process(vineyard_id, found)

    assert first == {"found": 3, "processed": 2, "cloudy": 1, "skipped": 0, "failed": 1}
    assert run_sql(
        "SELECT status, error IS NOT NULL FROM satellite_scans "
        "WHERE product_id = 'S2A_35TPN_20260715_1_L2A'"
    ) == [("failed", True)]

    history = api_client.get(
        f"/api/v1/vineyards/{vineyard_id}/vegetation", headers=headers
    ).json()
    # Снимок со сбоем в историю не попадает; облачный — без индексов
    assert [item["acquired_at"][:10] for item in history] == [
        "2026-06-03",
        "2026-08-29",
    ]
    cloudy, clear = history
    assert cloudy["ndvi"] is None
    assert cloudy["parcel_cloud_cover"] == 100
    assert clear["platform"] == "sentinel-2a"
    assert clear["parcel_cloud_cover"] == 0
    assert clear["ndvi"]["mean"] == pytest.approx(0.714, abs=1e-3)
    assert clear["ndre"]["mean"] == pytest.approx(0.5, abs=1e-3)

    # Повторный запуск: готовые снимки пропускаются, сбойный считается заново
    found[2] = make_scene(
        tmp_path,
        product_id="S2A_35TPN_20260715_1_L2A",
        acquired_at=datetime(2026, 7, 15, 9, 17, tzinfo=UTC),
    )
    second = process(vineyard_id, found)

    assert second == {
        "found": 3,
        "processed": 1,
        "cloudy": 0,
        "skipped": 2,
        "failed": 0,
    }
    assert run_sql(
        "SELECT count(*), count(*) FILTER (WHERE status = 'completed') "
        "FROM satellite_scans"
    ) == [(3, 3)]
    assert run_sql("SELECT count(*) FROM satellite_scans WHERE error IS NOT NULL") == [
        (0,)
    ]


def test_processing_missing_vineyard_fails(
    process: Callable[[str, list[EarthScene]], dict[str, int]],
) -> None:
    with pytest.raises(Exception, match="Участок не найден"):
        process("01a13ea2-b18e-77b7-aa02-2481b87c6b80", [])


def test_concurrent_task_saved_same_scene(
    api_client: TestClient,
    tmp_path: Path,
    process: Callable[[str, list[EarthScene]], dict[str, int]],
    run_sql: RunSql,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Пока задача читала каналы, тот же снимок сохранила параллельная задача:
    сохранение упирается в уникальность, и снимок считается пропущенным."""
    vineyard_id = create_vineyard(api_client, login(api_client))
    scene = make_scene(tmp_path)
    reading = vegetation_history._scene_indices

    def racing(found: EarthScene, geometry: dict[str, Any]) -> Any:
        result = reading(found, geometry)
        run_sql(
            "INSERT INTO satellite_scans (vineyard_id, product_id, acquired_at, "
            f"cloud_cover, status) VALUES ('{vineyard_id}', '{found.product_id}', "
            "now(), 0, 'completed')"
        )
        return result

    monkeypatch.setattr(vegetation_history, "_scene_indices", racing)

    result = process(vineyard_id, [scene])

    assert result == {
        "found": 1,
        "processed": 0,
        "cloudy": 0,
        "skipped": 1,
        "failed": 0,
    }
    assert run_sql("SELECT count(*) FROM satellite_scans") == [(1,)]


@pytest.fixture
def sent(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, list[str]]]:
    calls: list[tuple[str, list[str]]] = []

    def send_task(name: str, args: list[str]) -> SimpleNamespace:
        calls.append((name, args))
        return SimpleNamespace(id="task-1")

    monkeypatch.setattr(celery_app, "send_task", send_task)
    return calls


def test_refresh_starts_task_for_season(
    api_client: TestClient, sent: list[tuple[str, list[str]]]
) -> None:
    headers = login(api_client)
    vineyard_id = create_vineyard(api_client, headers)

    response = api_client.post(
        f"/api/v1/vineyards/{vineyard_id}/vegetation/refresh", headers=headers
    )

    assert response.status_code == 202
    assert response.json() == {"task_id": "task-1"}
    [(name, args)] = sent
    assert name == "satellite.process_vineyard"
    today = datetime.now(UTC).date()
    assert args[0] == vineyard_id
    assert args[1].startswith(vineyard_routes.season_start(today).isoformat())
    assert args[2].startswith(today.isoformat())


@pytest.mark.parametrize(
    ("params", "detail"),
    [
        ({"start": "2026-08-01", "end": "2026-07-01"}, "Начало периода"),
        ({"start": "2024-01-01", "end": "2026-01-01"}, "не больше года"),
    ],
)
def test_refresh_rejects_bad_period(
    api_client: TestClient,
    sent: list[tuple[str, list[str]]],
    params: dict[str, str],
    detail: str,
) -> None:
    headers = login(api_client)
    vineyard_id = create_vineyard(api_client, headers)

    response = api_client.post(
        f"/api/v1/vineyards/{vineyard_id}/vegetation/refresh",
        headers=headers,
        params=params,
    )

    assert response.status_code == 400
    assert detail in response.json()["detail"]
    assert sent == []


def test_vegetation_of_foreign_vineyard_looks_missing(
    api_client: TestClient, sent: list[tuple[str, list[str]]]
) -> None:
    vineyard_id = create_vineyard(api_client, login(api_client))
    stranger = login(api_client, "stranger@example.md")

    for method, path in (("post", "/refresh"), ("get", "")):
        response = api_client.request(
            method,
            f"/api/v1/vineyards/{vineyard_id}/vegetation{path}",
            headers=stranger,
        )
        assert response.status_code == 404
    assert sent == []


@pytest.mark.parametrize(
    ("today", "expected"),
    [
        (date(2026, 10, 16), date(2026, 3, 1)),
        (date(2026, 3, 1), date(2026, 3, 1)),
        (date(2027, 1, 20), date(2026, 3, 1)),
    ],
)
def test_season_starts_in_march(today: date, expected: date) -> None:
    assert vineyard_routes.season_start(today) == expected
