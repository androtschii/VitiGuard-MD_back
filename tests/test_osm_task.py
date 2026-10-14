import pytest

from app.core.config import Settings
from app.tasks import osm


def test_task_runs_import_with_current_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = Settings(_env_file=None, environment="test")
    calls: list[Settings] = []

    async def fake_run_import(received: Settings) -> dict[str, int]:
        calls.append(received)
        return {"imported": 1, "deleted": 0, "skipped": 0}

    monkeypatch.setattr(osm, "get_settings", lambda: settings)
    monkeypatch.setattr(osm, "run_import", fake_run_import)

    assert osm.import_vineyards() == {"imported": 1, "deleted": 0, "skipped": 0}
    assert calls == [settings]
