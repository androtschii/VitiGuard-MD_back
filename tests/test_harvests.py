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


def harvests_url(client: TestClient, headers: dict[str, str]) -> str:
    response = client.post(
        "/api/v1/vineyards",
        headers=headers,
        json={"name": "Криково", "geometry": POLYGON},
    )
    return f"/api/v1/vineyards/{response.json()['id']}/harvests"


def test_same_season_replaces_and_newest_first(api_client: TestClient) -> None:
    headers = sign_in(api_client, CREDENTIALS["email"])
    url = harvests_url(api_client, headers)
    year = local_today().year

    last = {"season": year - 1, "actual_kg": 900}
    assert api_client.post(url, headers=headers, json=last).status_code == 201
    made = api_client.post(
        url, headers=headers, json={"season": year, "estimated_kg": 1000}
    ).json()
    updated = api_client.post(
        url,
        headers=headers,
        json={"season": year, "estimated_kg": 1000, "actual_kg": 850, "sugar_brix": 22},
    ).json()

    assert updated["id"] == made["id"]
    rows = api_client.get(url, headers=headers).json()
    assert [row["season"] for row in rows] == [year, year - 1]
    assert rows[0]["actual_kg"] == 850
    one = f"{url}/{made['id']}"
    assert api_client.delete(one, headers=headers).status_code == 204
    assert api_client.delete(one, headers=headers).status_code == 404


def test_rejects_wrong_values_and_other_owner(api_client: TestClient) -> None:
    headers = sign_in(api_client, CREDENTIALS["email"])
    url = harvests_url(api_client, headers)
    year = local_today().year

    empty = {"season": year}
    assert api_client.post(url, headers=headers, json=empty).status_code == 400
    future = {"season": year + 1, "estimated_kg": 1}
    assert api_client.post(url, headers=headers, json=future).status_code == 400
    negative = {"season": year, "actual_kg": -1}
    assert api_client.post(url, headers=headers, json=negative).status_code == 422
    other = sign_in(api_client, "other@example.md")
    assert api_client.get(url, headers=other).status_code == 404
    good = {"season": year, "actual_kg": 1}
    assert api_client.post(url, headers=other, json=good).status_code == 404
