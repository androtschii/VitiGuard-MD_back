import asyncio
from collections.abc import Callable

import pytest
from fastapi.testclient import TestClient

from app.cli import load_seed
from app.core.config import Settings

pytestmark = pytest.mark.integration


def test_seed_creates_users_once(
    api_client: TestClient, migrated_database_url: str
) -> None:
    settings = Settings(
        _env_file=None,
        database_url=migrated_database_url,  # type: ignore[arg-type]
    )

    assert asyncio.run(load_seed.run(settings, 2)) == 2
    # Повторный запуск дописывает только недостающих
    assert asyncio.run(load_seed.run(settings, 3)) == 1

    token = api_client.post(
        "/api/v1/auth/login",
        json={"email": load_seed.email(2), "password": load_seed.PASSWORD},
    ).json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    vineyards = api_client.get("/api/v1/vineyards", headers=headers).json()
    assert vineyards["total"] == load_seed.VINEYARDS_PER_USER
    assert all(
        item["area_ha"] == pytest.approx(4, abs=0.6) for item in vineyards["items"]
    )
    today = api_client.get("/api/v1/risks/today", headers=headers).json()
    assert len(today) == load_seed.VINEYARDS_PER_USER


def test_main_reports_created(
    migrated_database_url: str,
    run_sql: Callable[[str], list[tuple[object, ...]]],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    settings = Settings(
        _env_file=None,
        database_url=migrated_database_url,  # type: ignore[arg-type]
    )
    monkeypatch.setattr(load_seed, "get_settings", lambda: settings)

    try:
        load_seed.main(["--users", "1"])
    finally:
        # Без api_client таблицу пользователей после теста никто не очистит
        run_sql("TRUNCATE users CASCADE")

    assert "Создано пользователей: 1" in capsys.readouterr().out
