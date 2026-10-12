import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.integration

URL = "/api/v1/varieties"
MISSING = f"{URL}/01900000-0000-7000-8000-000000000000"


def login_as(api_client: TestClient, email: str, role: str) -> dict[str, str]:
    credentials = {"email": email, "password": "secret123"}
    api_client.post(
        "/api/v1/auth/register",
        json={**credentials, "full_name": "Тест", "role": role},
    )
    token = api_client.post("/api/v1/auth/login", json=credentials).json()[
        "access_token"
    ]
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def grower(api_client: TestClient) -> dict[str, str]:
    return login_as(api_client, "grower@example.md", "user")


@pytest.fixture
def agronomist(api_client: TestClient) -> dict[str, str]:
    return login_as(api_client, "agronomist@example.md", "agronomist")


def test_moldovan_varieties_are_preloaded(
    api_client: TestClient, grower: dict[str, str]
) -> None:
    varieties = api_client.get(URL, headers=grower).json()

    names = [variety["name"] for variety in varieties]
    assert names == sorted(names)
    assert {"Fetească Neagră", "Rara Neagră", "Merlot", "Traminer Roz"} <= set(names)
    feteasca = next(v for v in varieties if v["name"] == "Fetească Neagră")
    assert feteasca["berry_color"] == "red"
    # Восприимчивость не выдумывается: её заполняет агроном
    assert feteasca["downy_mildew_susceptibility"] is None


def test_agronomist_manages_varieties(
    api_client: TestClient, agronomist: dict[str, str], grower: dict[str, str]
) -> None:
    created = api_client.post(
        URL,
        headers=agronomist,
        json={
            "name": "  Viorica  ",
            "berry_color": "white",
            "downy_mildew_susceptibility": "low",
        },
    )
    assert created.status_code == 201
    variety_id = created.json()["id"]
    assert created.json()["name"] == "Viorica"

    updated = api_client.patch(
        f"{URL}/{variety_id}",
        headers=agronomist,
        json={
            "downy_mildew_susceptibility": None,
            "powdery_mildew_susceptibility": "medium",
        },
    )
    assert updated.status_code == 200
    assert updated.json()["downy_mildew_susceptibility"] is None
    assert updated.json()["powdery_mildew_susceptibility"] == "medium"
    assert updated.json()["berry_color"] == "white"

    renamed = api_client.patch(
        f"{URL}/{variety_id}", headers=agronomist, json={"name": "Viorica Nouă"}
    )
    assert renamed.json()["name"] == "Viorica Nouă"

    read = api_client.get(f"{URL}/{variety_id}", headers=grower)
    assert read.json()["powdery_mildew_susceptibility"] == "medium"

    deleted = api_client.delete(f"{URL}/{variety_id}", headers=agronomist)
    assert deleted.status_code == 204
    assert api_client.get(f"{URL}/{variety_id}", headers=grower).status_code == 404


def test_name_is_unique_regardless_of_case(
    api_client: TestClient, agronomist: dict[str, str]
) -> None:
    response = api_client.post(
        URL, headers=agronomist, json={"name": "merlot", "berry_color": "red"}
    )

    assert response.status_code == 409
    assert response.json()["detail"] == "Сорт с таким названием уже есть"


def test_grower_cannot_change_reference(
    api_client: TestClient, grower: dict[str, str]
) -> None:
    merlot = next(
        v for v in api_client.get(URL, headers=grower).json() if v["name"] == "Merlot"
    )

    assert (
        api_client.post(
            URL, headers=grower, json={"name": "Свой", "berry_color": "red"}
        ).status_code
        == 403
    )
    assert (
        api_client.patch(
            f"{URL}/{merlot['id']}", headers=grower, json={"berry_color": "white"}
        ).status_code
        == 403
    )
    assert api_client.delete(f"{URL}/{merlot['id']}", headers=grower).status_code == 403


@pytest.mark.parametrize(
    "body",
    [
        {"name": None},
        {"berry_color": None},
        {"berry_color": "blue"},
        {"name": ""},
        {"downy_mildew_susceptibility": "extreme"},
    ],
)
def test_update_validation(
    api_client: TestClient, agronomist: dict[str, str], body: dict[str, object]
) -> None:
    merlot = next(
        v
        for v in api_client.get(URL, headers=agronomist).json()
        if v["name"] == "Merlot"
    )

    response = api_client.patch(f"{URL}/{merlot['id']}", headers=agronomist, json=body)

    assert response.status_code == 422


def test_missing_variety(api_client: TestClient, agronomist: dict[str, str]) -> None:
    assert api_client.get(MISSING, headers=agronomist).status_code == 404
    assert api_client.patch(MISSING, headers=agronomist, json={}).status_code == 404
    assert api_client.delete(MISSING, headers=agronomist).status_code == 404


def test_requires_login(api_client: TestClient) -> None:
    assert api_client.get(URL).status_code == 401
