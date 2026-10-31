from datetime import date, timedelta

import pytest
from fastapi.testclient import TestClient

from app.models.enums import Fungicide
from app.models.treatment import Treatment
from app.services.risk import local_today
from app.services.treatments import harvest_allowed_from

CREDENTIALS = {"email": "grower@example.md", "password": "secret123"}
POLYGON = {
    "type": "MultiPolygon",
    "coordinates": [
        [[[28.8, 47.0], [28.81, 47.0], [28.81, 47.01], [28.8, 47.01], [28.8, 47.0]]]
    ],
}


def sign_in(client: TestClient, email: str) -> dict[str, str]:
    body = {**CREDENTIALS, "email": email}
    client.post("/api/v1/auth/register", json={**body, "full_name": "Ион"})
    token = client.post("/api/v1/auth/login", json=body).json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def events_url(client: TestClient, headers: dict[str, str]) -> str:
    response = client.post(
        "/api/v1/vineyards",
        headers=headers,
        json={"name": "Криково", "geometry": POLYGON},
    )
    return f"/api/v1/vineyards/{response.json()['id']}/weather-events"


@pytest.mark.integration
def test_event_lifecycle(api_client: TestClient) -> None:
    headers = sign_in(api_client, CREDENTIALS["email"])
    url = events_url(api_client, headers)
    body = {
        "occurred_on": local_today().isoformat(),
        "kind": "hail",
        "severity": 3,
        "note": "Побиты верхние ряды",
    }

    created = api_client.post(url, headers=headers, json=body)

    assert created.status_code == 201
    assert api_client.get(url, headers=headers).json()[0]["note"] == body["note"]
    one = f"{url}/{created.json()['id']}"
    assert api_client.delete(one, headers=headers).status_code == 204
    assert api_client.delete(one, headers=headers).status_code == 404


@pytest.mark.integration
def test_rejects_wrong_values_and_other_owner(api_client: TestClient) -> None:
    headers = sign_in(api_client, CREDENTIALS["email"])
    url = events_url(api_client, headers)
    today = local_today()
    good = {"occurred_on": today.isoformat(), "kind": "frost", "severity": 1}

    for wrong in (
        {**good, "kind": "meteor"},
        {**good, "severity": 4},
    ):
        assert api_client.post(url, headers=headers, json=wrong).status_code == 422
    future = {**good, "occurred_on": (today + timedelta(days=1)).isoformat()}
    assert api_client.post(url, headers=headers, json=future).status_code == 400
    other = sign_in(api_client, "other@example.md")
    assert api_client.get(url, headers=other).status_code == 404
    assert api_client.post(url, headers=other, json=good).status_code == 404


def treated(day: date, waiting: int | None) -> Treatment:
    return Treatment(applied_on=day, fungicide=Fungicide.CONTACT, waiting_days=waiting)


def test_harvest_date_is_latest_waiting_period() -> None:
    today = date(2026, 6, 10)
    items = [treated(date(2026, 6, 1), 30), treated(date(2026, 6, 8), 7)]

    assert harvest_allowed_from(items, today) == date(2026, 7, 1)
    assert harvest_allowed_from([treated(today, None)], today) is None
    assert harvest_allowed_from([treated(date(2026, 6, 1), 3)], today) is None
    assert harvest_allowed_from([], today) is None
