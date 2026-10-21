from collections.abc import Callable
from datetime import date, timedelta

import pytest
from fastapi.testclient import TestClient

from app.models.enums import Disease, Fungicide, RiskLevel
from app.models.treatment import Treatment
from app.models.weather import DiseaseRisk
from app.services.agromodels import DailyWeather
from app.services.risk import local_today
from app.services.treatments import Action, plan, protection_end

TODAY = date(2026, 6, 10)
RunSql = Callable[[str], list[tuple[object, ...]]]


def day(offset: int) -> date:
    return TODAY + timedelta(days=offset)


def risk(method: str, offset: int, level: RiskLevel = RiskLevel.HIGH) -> DiseaseRisk:
    disease = (
        Disease.POWDERY_MILDEW if method == "gubler_thomas" else Disease.DOWNY_MILDEW
    )
    return DiseaseRisk(
        disease=disease, method=method, target_date=day(offset), score=1, level=level
    )


def treated(fungicide: Fungicide, offset: int) -> Treatment:
    return Treatment(applied_on=day(offset), fungicide=fungicide)


def weather(
    rain: dict[int, float] | None = None, heat: dict[int, float] | None = None
) -> list[DailyWeather]:
    rain, heat = rain or {}, heat or {}
    return [
        DailyWeather(
            day=day(offset),
            mean_temp=20,
            min_temp=14,
            max_temp=heat.get(offset, 26),
            precipitation_mm=rain.get(offset, 0),
            mean_humidity=70,
            longest_mild_run=8,
            hours_above_35=0,
        )
        for offset in range(-15, 8)
    ]


def test_forecast_infection_asks_for_protection_before_rain() -> None:
    result = plan(TODAY, [risk("three_tens", 2)], [], weather())

    [item] = result.recommendations
    assert item.action is Action.PROTECT
    assert item.fungicide is Fungicide.CONTACT
    assert (item.due_by, item.risk_day) == (day(1), day(2))
    assert item.reason == "downy_infection_forecast"
    assert result.protected_until == {
        Disease.DOWNY_MILDEW: None,
        Disease.POWDERY_MILDEW: None,
    }


def test_rain_today_leaves_only_systemic() -> None:
    [item] = plan(TODAY, [risk("three_tens", 0)], [], weather()).recommendations

    assert item.fungicide is Fungicide.SYSTEMIC
    assert item.due_by == TODAY


def test_fresh_contact_treatment_protects() -> None:
    result = plan(
        TODAY,
        [risk("three_tens", 2), risk("three_tens", 3, RiskLevel.LOW)],
        [treated(Fungicide.CONTACT, -1)],
        weather(),
    )

    assert result.recommendations == []
    # Восемь дней защиты, включая день обработки
    assert result.protected_until[Disease.DOWNY_MILDEW] == day(6)
    # Медь от оидиума не защищает
    assert result.protected_until[Disease.POWDERY_MILDEW] is None


def test_heavy_rain_washes_contact_treatment_off() -> None:
    contact = treated(Fungicide.CONTACT, -3)
    days = {item.day: item for item in weather(rain={-2: 8, 0: 15})}

    # 8 + 15 мм к сегодняшнему дню — больше 20: сегодня защиты уже нет
    assert protection_end(contact, Disease.DOWNY_MILDEW, days) == day(-1)
    result = plan(
        TODAY, [risk("three_tens", 2)], [contact], weather(rain={-2: 8, 0: 15})
    )
    assert [item.reason for item in result.recommendations] == [
        "downy_infection_forecast"
    ]


def test_rain_does_not_wash_systemic_off() -> None:
    systemic = treated(Fungicide.SYSTEMIC, -3)
    days = {item.day: item for item in weather(rain={-2: 40})}

    assert protection_end(systemic, Disease.DOWNY_MILDEW, days) == day(8)


def test_missed_infection_needs_cure_within_three_days() -> None:
    [item] = plan(TODAY, [risk("three_tens", -2)], [], weather()).recommendations

    assert item.action is Action.CURE
    assert item.fungicide is Fungicide.SYSTEMIC
    assert (item.risk_day, item.due_by) == (day(-2), day(1))


def test_cure_window_closes() -> None:
    # Заражение четыре дня назад: лечить поздно, остаётся ждать пятен
    assert plan(TODAY, [risk("three_tens", -4)], [], weather()).recommendations == []


def test_infection_on_protected_day_needs_nothing() -> None:
    result = plan(
        TODAY,
        [risk("three_tens", -2)],
        [treated(Fungicide.SYSTEMIC, -5)],
        weather(),
    )

    assert result.recommendations == []


def test_expected_symptoms_ask_for_inspection() -> None:
    [item] = plan(TODAY, [risk("goidanich", 3)], [], weather()).recommendations

    assert item.action is Action.INSPECT
    assert item.fungicide is None
    assert item.due_by == day(3)


def test_powdery_high_asks_for_sulfur() -> None:
    [item] = plan(TODAY, [risk("gubler_thomas", 1)], [], weather()).recommendations

    assert (item.disease, item.fungicide) == (Disease.POWDERY_MILDEW, Fungicide.SULFUR)
    assert item.reason == "powdery_high"


def test_heat_swaps_sulfur_for_systemic() -> None:
    [item] = plan(
        TODAY, [risk("gubler_thomas", 1)], [], weather(heat={2: 33})
    ).recommendations

    assert item.fungicide is Fungicide.SYSTEMIC
    assert item.reason == "powdery_high_hot"


def test_sulfur_protects_from_powdery_only_for_a_week() -> None:
    sulfur = treated(Fungicide.SULFUR, -6)
    result = plan(
        TODAY,
        [risk("gubler_thomas", 0), risk("gubler_thomas", 1)],
        [sulfur],
        weather(),
    )

    assert result.protected_until[Disease.POWDERY_MILDEW] == TODAY
    assert [(item.due_by, item.fungicide) for item in result.recommendations] == [
        (day(1), Fungicide.SULFUR)
    ]
    assert protection_end(sulfur, Disease.DOWNY_MILDEW, {}) is None


def test_recommendations_are_sorted_by_deadline() -> None:
    result = plan(
        TODAY,
        [risk("gubler_thomas", 4), risk("three_tens", 2), risk("three_tens", -1)],
        [],
        weather(),
    )

    assert [item.due_by for item in result.recommendations] == [
        day(1),
        day(2),
        day(4),
    ]


def login(client: TestClient, email: str = "grower@example.md") -> dict[str, str]:
    credentials = {"email": email, "password": "secret123"}
    client.post("/api/v1/auth/register", json={**credentials, "full_name": "Ион"})
    token = client.post("/api/v1/auth/login", json=credentials).json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def create_vineyard(
    client: TestClient, headers: dict[str, str], west: float = 28.86
) -> str:
    ring = [[west, 47.14], [west + 0.002, 47.14], [west + 0.002, 47.141], [west, 47.14]]
    vineyard_id: str = client.post(
        "/api/v1/vineyards",
        headers=headers,
        json={
            "name": "Криково",
            "geometry": {"type": "Polygon", "coordinates": [ring]},
        },
    ).json()["id"]
    return vineyard_id


@pytest.mark.integration
def test_treatments_api(api_client: TestClient, run_sql: RunSql) -> None:
    headers = login(api_client)
    vineyard = create_vineyard(api_client, headers)
    url = f"/api/v1/vineyards/{vineyard}/treatments"
    today = local_today()
    # Заражение милдью по прогнозу послезавтра
    run_sql(
        "INSERT INTO disease_risks (vineyard_id, disease, target_date, method, "
        f"score, level) VALUES ('{vineyard}', 'downy_mildew', "
        f"'{today + timedelta(days=2)}', 'three_tens', 1, 'high')"
    )

    before = api_client.get(f"{url}/plan", headers=headers).json()
    assert before["today"] == today.isoformat()
    assert [item["reason"] for item in before["recommendations"]] == [
        "downy_infection_forecast"
    ]
    assert before["recommendations"][0]["fungicide"] == "contact"

    created = api_client.post(
        url,
        headers=headers,
        json={
            "applied_on": today.isoformat(),
            "fungicide": "contact",
            "product": "Купроксат",
        },
    )
    assert created.status_code == 201
    assert created.json()["product"] == "Купроксат"
    after = api_client.get(f"{url}/plan", headers=headers).json()
    assert after["recommendations"] == []
    assert (
        after["protected_until"]["downy_mildew"]
        == (today + timedelta(days=7)).isoformat()
    )

    future = api_client.post(
        url,
        headers=headers,
        json={
            "applied_on": (today + timedelta(days=1)).isoformat(),
            "fungicide": "sulfur",
        },
    )
    assert future.status_code == 400
    assert [
        item["fungicide"] for item in api_client.get(url, headers=headers).json()
    ] == ["contact"]

    treatment = created.json()["id"]
    stranger = login(api_client, "stranger@example.md")
    assert api_client.get(url, headers=stranger).status_code == 404
    assert api_client.delete(f"{url}/{treatment}", headers=stranger).status_code == 404
    other = create_vineyard(api_client, headers, 28.87)
    assert (
        api_client.delete(
            f"/api/v1/vineyards/{other}/treatments/{treatment}", headers=headers
        ).status_code
        == 404
    )
    assert api_client.delete(f"{url}/{treatment}", headers=headers).status_code == 204
    assert api_client.get(url, headers=headers).json() == []
    assert api_client.get(f"{url}/plan").status_code == 401
