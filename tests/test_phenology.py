from datetime import timedelta

import pytest
from fastapi.testclient import TestClient

from app.services.risk import local_today

pytestmark = pytest.mark.integration

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


def phenology_url(client: TestClient, headers: dict[str, str]) -> str:
    response = client.post(
        "/api/v1/vineyards",
        headers=headers,
        json={"name": "Криково", "geometry": POLYGON},
    )
    return f"/api/v1/vineyards/{response.json()['id']}/phenology"


def test_same_day_replaces_and_list_is_newest_first(api_client: TestClient) -> None:
    headers = sign_in(api_client, CREDENTIALS["email"])
    url = phenology_url(api_client, headers)
    today = local_today()
    yesterday = today - timedelta(days=1)

    first = {"observed_on": yesterday.isoformat(), "stage": 19}
    assert api_client.post(url, headers=headers, json=first).status_code == 201
    second = {"observed_on": today.isoformat(), "stage": 23}
    made = api_client.post(url, headers=headers, json=second).json()
    replaced = api_client.post(
        url, headers=headers, json={**second, "stage": 27}
    ).json()

    assert replaced["id"] == made["id"]
    rows = api_client.get(url, headers=headers).json()
    assert [row["stage"] for row in rows] == [27, 19]
    assert api_client.delete(f"{url}/{made['id']}", headers=headers).status_code == 204
    assert api_client.delete(f"{url}/{made['id']}", headers=headers).status_code == 404


def test_rejects_unknown_stage_and_future_day(api_client: TestClient) -> None:
    headers = sign_in(api_client, CREDENTIALS["email"])
    url = phenology_url(api_client, headers)
    today = local_today()

    unknown = {"observed_on": today.isoformat(), "stage": 5}
    assert api_client.post(url, headers=headers, json=unknown).status_code == 422
    future = {"observed_on": (today + timedelta(days=1)).isoformat(), "stage": 4}
    assert api_client.post(url, headers=headers, json=future).status_code == 400


def test_other_owner_sees_nothing(api_client: TestClient) -> None:
    headers = sign_in(api_client, CREDENTIALS["email"])
    url = phenology_url(api_client, headers)
    other = sign_in(api_client, "other@example.md")

    assert api_client.get(url, headers=other).status_code == 404
    body = {"observed_on": local_today().isoformat(), "stage": 4}
    assert api_client.post(url, headers=other, json=body).status_code == 404
