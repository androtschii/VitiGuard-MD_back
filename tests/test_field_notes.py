import uuid

import pytest
from diagnosis_model import photo
from fastapi.testclient import TestClient
from memory_storage import MemoryStorage

from app.services.risk import local_today

pytestmark = pytest.mark.integration

CREDENTIALS = {"email": "grower@example.md", "password": "secret123"}
POLYGON = {
    "type": "MultiPolygon",
    "coordinates": [
        [[[28.8, 47.0], [28.81, 47.0], [28.81, 47.01], [28.8, 47.01], [28.8, 47.0]]]
    ],
}


@pytest.fixture
def storage(api_client: TestClient) -> MemoryStorage:
    memory = MemoryStorage()
    api_client.app.state.storage = memory  # type: ignore[attr-defined]
    return memory


def sign_in(client: TestClient, email: str) -> dict[str, str]:
    body = {**CREDENTIALS, "email": email}
    client.post("/api/v1/auth/register", json={**body, "full_name": "Ион"})
    token = client.post("/api/v1/auth/login", json=body).json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def plot(client: TestClient, headers: dict[str, str]) -> str:
    response = client.post(
        "/api/v1/vineyards",
        headers=headers,
        json={"name": "Криково", "geometry": POLYGON},
    )
    assert response.status_code == 201
    return str(response.json()["id"])


def add_note(
    client: TestClient,
    headers: dict[str, str],
    vineyard_id: str,
    data: dict[str, str],
    image: bytes | None = None,
) -> dict[str, object]:
    response = client.post(
        f"/api/v1/vineyards/{vineyard_id}/notes",
        headers=headers,
        data={"noted_on": local_today().isoformat(), **data},
        files={"photo": ("leaf.jpg", image, "image/jpeg")} if image else None,
    )
    return {"status": response.status_code, **response.json()}


def test_note_with_photo_and_point(
    api_client: TestClient, storage: MemoryStorage
) -> None:
    headers = sign_in(api_client, CREDENTIALS["email"])
    vineyard_id = plot(api_client, headers)

    note = add_note(
        api_client,
        headers,
        vineyard_id,
        {"text": " Пятна на нижнем ярусе ", "latitude": "47.1", "longitude": "28.8"},
        photo((255, 0, 0)),
    )

    assert note["status"] == 201
    assert note["text"] == "Пятна на нижнем ярусе"
    assert (note["latitude"], note["longitude"]) == (47.1, 28.8)
    assert note["has_photo"] is True
    url = f"/api/v1/vineyards/{vineyard_id}/notes/{note['id']}"
    image = api_client.get(f"{url}/photo", headers=headers)
    assert image.status_code == 200
    assert image.headers["content-type"] == "image/jpeg"
    assert api_client.delete(url, headers=headers).status_code == 204
    assert api_client.get(f"{url}/photo", headers=headers).status_code == 404
    assert not storage.files


def test_note_without_photo_has_no_photo_url(
    api_client: TestClient, storage: MemoryStorage
) -> None:
    headers = sign_in(api_client, CREDENTIALS["email"])
    vineyard_id = plot(api_client, headers)

    note = add_note(api_client, headers, vineyard_id, {"text": "Осмотр"})

    assert note["has_photo"] is False
    url = f"/api/v1/vineyards/{vineyard_id}/notes/{note['id']}"
    assert api_client.get(f"{url}/photo", headers=headers).status_code == 404
    assert api_client.delete(url, headers=headers).status_code == 204


def test_note_needs_text_or_photo_and_valid_point(
    api_client: TestClient, storage: MemoryStorage
) -> None:
    headers = sign_in(api_client, CREDENTIALS["email"])
    vineyard_id = plot(api_client, headers)

    assert add_note(api_client, headers, vineyard_id, {"text": "  "})["status"] == 400
    half = {"text": "x", "latitude": "47"}
    assert add_note(api_client, headers, vineyard_id, half)["status"] == 400
    wrong = {"text": "x", "latitude": "91", "longitude": "28"}
    assert add_note(api_client, headers, vineyard_id, wrong)["status"] == 422
    future = {"text": "x", "noted_on": "2999-01-01"}
    assert add_note(api_client, headers, vineyard_id, future)["status"] == 400
    bad = add_note(api_client, headers, vineyard_id, {}, b"not an image")
    assert bad["status"] == 415
    assert not storage.files


def test_repeat_from_queue_does_not_duplicate(
    api_client: TestClient, storage: MemoryStorage
) -> None:
    headers = sign_in(api_client, CREDENTIALS["email"])
    vineyard_id = plot(api_client, headers)
    data = {"text": "Осмотр", "client_id": str(uuid.uuid4())}

    first = add_note(api_client, headers, vineyard_id, data)
    second = add_note(api_client, headers, vineyard_id, data)

    assert first["id"] == second["id"]
    notes = api_client.get(f"/api/v1/vineyards/{vineyard_id}/notes", headers=headers)
    assert len(notes.json()) == 1


def test_other_owner_sees_nothing(
    api_client: TestClient, storage: MemoryStorage
) -> None:
    headers = sign_in(api_client, CREDENTIALS["email"])
    vineyard_id = plot(api_client, headers)
    note = add_note(api_client, headers, vineyard_id, {"text": "Моё"})
    other = sign_in(api_client, "other@example.md")

    url = f"/api/v1/vineyards/{vineyard_id}/notes"
    assert api_client.get(url, headers=other).status_code == 404
    assert api_client.delete(f"{url}/{note['id']}", headers=other).status_code == 404
    assert add_note(api_client, other, vineyard_id, {"text": "x"})["status"] == 404


def test_deleting_vineyard_removes_note_photos(
    api_client: TestClient, storage: MemoryStorage
) -> None:
    headers = sign_in(api_client, CREDENTIALS["email"])
    vineyard_id = plot(api_client, headers)
    add_note(api_client, headers, vineyard_id, {}, photo((0, 255, 0)))
    assert storage.files

    response = api_client.delete(f"/api/v1/vineyards/{vineyard_id}", headers=headers)

    assert response.status_code == 204
    assert not storage.files
