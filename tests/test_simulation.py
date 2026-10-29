import asyncio
from collections import defaultdict
from datetime import date
from itertools import pairwise

import pytest
from fastapi.testclient import TestClient

from app.cli import simulate as simulate_cli
from app.core.config import Settings
from app.services.agromodels import daily_weather, season_risks
from app.services.anomaly import Observation, detect_anomalies
from app.services.simulation import (
    Scenario,
    mean_temperature,
    simulate_scans,
    simulate_weather,
)


def infections(scenario: Scenario, seed: int) -> int:
    risks = season_risks(simulate_weather(2025, scenario, seed))
    return sum(1 for r in risks if r.method == "three_tens" and r.score == 1)


def anomalies(scenario: Scenario, seed: int) -> list[tuple[int, str]]:
    scans = simulate_scans(2025, scenario, seed)
    found = detect_anomalies(
        [Observation(s.acquired_at, s.ndvi) for s in scans if s.ndvi is not None], []
    )
    return [(a.acquired_at.month, a.level) for a in found]


def test_weather_is_hourly_local_season_and_reproducible() -> None:
    hours = simulate_weather(2025, Scenario.normal, 7)

    days = daily_weather(hours)
    # С 1 марта по 30 сентября, каждые сутки полные (перевод часов учтён)
    assert (days[0].day, days[-1].day, len(days)) == (
        date(2025, 3, 1),
        date(2025, 9, 30),
        214,
    )
    assert hours == simulate_weather(2025, Scenario.normal, 7)
    assert hours != simulate_weather(2025, Scenario.normal, 8)
    july = [d.mean_temp for d in days if d.day.month == 7]
    assert 18 < sum(july) / len(july) < 28
    assert all(
        h.precipitation_mm >= 0 and 25 <= h.relative_humidity <= 100 for h in hours
    )


def test_climate_curve() -> None:
    assert mean_temperature(date(2025, 3, 1)) < 6
    assert mean_temperature(date(2025, 7, 25)) > 23


def test_wet_season_brings_more_downy_mildew_than_dry() -> None:
    seeds = range(1, 6)
    dry = sum(infections(Scenario.dry, seed) for seed in seeds)
    wet = sum(infections(Scenario.wet, seed) for seed in seeds)

    assert wet > 2 * dry


def test_only_outbreak_has_confirmed_ndvi_drop() -> None:
    by_scenario: dict[Scenario, list[list[tuple[int, str]]]] = defaultdict(list)
    for scenario in Scenario:
        for seed in range(1, 11):
            by_scenario[scenario].append(anomalies(scenario, seed))

    # Очаг виден в каждом сезоне со вспышкой: подтверждённая просадка летом
    assert all(
        any(level == "confirmed" and month in (7, 8) for month, level in found)
        for found in by_scenario[Scenario.outbreak]
    )
    # В здоровых сезонах подтверждённых просадок нет; на быстром весеннем
    # росте метод изредка даёт одиночное «подозрение» — это его свойство
    for scenario in (Scenario.dry, Scenario.normal, Scenario.wet):
        for found in by_scenario[scenario]:
            assert all(level == "suspected" for _, level in found)


def test_scans_every_five_days_some_under_clouds() -> None:
    scans = simulate_scans(2025, Scenario.normal, 3)

    assert scans[0].acquired_at.date() == date(2025, 4, 1)
    gaps = {(b.acquired_at - a.acquired_at).days for a, b in pairwise(scans)}
    assert gaps == {5}
    cloudy = [s for s in scans if s.ndvi is None]
    assert cloudy
    assert all(s.parcel_cloud_cover == 100 and s.ndre is None for s in cloudy)


def login(client: TestClient, email: str) -> dict[str, str]:
    token = client.post(
        "/api/v1/auth/login", json={"email": email, "password": "secret123"}
    ).json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


@pytest.mark.integration
def test_simulated_season_is_visible_in_the_app(
    api_client: TestClient, migrated_database_url: str
) -> None:
    settings = Settings(
        _env_file=None,
        database_url=migrated_database_url,  # type: ignore[arg-type]
    )

    summary = asyncio.run(
        simulate_cli.run(
            settings, "Student@Example.md", "secret123", Scenario.outbreak, 2025, 7
        )
    )

    # Весной часы переводят вперёд: в сезоне на один час меньше 214 суток
    assert summary.hours == 214 * 24 - 1
    assert summary.infections > 0
    assert summary.anomalies >= 2
    headers = login(api_client, "student@example.md")
    url = f"/api/v1/vineyards/{summary.vineyard_id}"
    vineyard = api_client.get(url, headers=headers).json()
    assert vineyard["name"] == "Учебный участок: outbreak, 2025, зерно 7"
    history = api_client.get(f"{url}/vegetation", headers=headers).json()
    assert len(history) == summary.scans
    found = api_client.get(f"{url}/vegetation/anomalies", headers=headers).json()
    assert {item["acquired_at"][:7] for item in found} <= {"2025-07", "2025-08"}

    # Второй участок той же учётной записи: пароль больше не нужен
    again = asyncio.run(
        simulate_cli.run(settings, "student@example.md", None, Scenario.dry, 2025, 8)
    )
    total = api_client.get("/api/v1/vineyards", headers=headers).json()["total"]
    assert total == 2
    assert again.vineyard_id != summary.vineyard_id


@pytest.mark.integration
def test_new_account_needs_password(migrated_database_url: str) -> None:
    settings = Settings(
        _env_file=None,
        database_url=migrated_database_url,  # type: ignore[arg-type]
    )

    with pytest.raises(SystemExit, match="пароль"):
        asyncio.run(
            simulate_cli.run(settings, "nobody@example.md", None, Scenario.dry, 2025, 1)
        )


@pytest.mark.integration
def test_main_prints_summary(
    api_client: TestClient,
    migrated_database_url: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    settings = Settings(
        _env_file=None,
        database_url=migrated_database_url,  # type: ignore[arg-type]
    )
    monkeypatch.setattr(simulate_cli, "get_settings", lambda: settings)

    simulate_cli.main(
        ["--email", "lab@example.md", "--password", "secret123", "--scenario", "wet"]
    )

    printed = capsys.readouterr().out
    assert printed.startswith("Участок ")
    assert "заражений милдью" in printed
