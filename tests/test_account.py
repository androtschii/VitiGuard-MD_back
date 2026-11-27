import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

import pytest
from conftest import set_role
from fastapi.testclient import TestClient
from memory_storage import MemoryStorage

from app.models.enums import Disease, Fungicide, RiskLevel
from app.services.diagnosis_history import overlay_key, thumb_key
from app.services.risk import local_today

pytestmark = pytest.mark.integration

RunSql = Callable[[str], list[tuple[object, ...]]]
URL_DELETE = "/api/v1/users/me"
URL_EXPORT = "/api/v1/users/me/export"
PASSWORD = "secret123"
SQUARE: dict[str, Any] = {
    "type": "Polygon",
    "coordinates": [
        [[28.86, 47.14], [28.862, 47.14], [28.862, 47.141], [28.86, 47.14]]
    ],
}


def register_and_login(
    client: TestClient, email: str, full_name: str = "Тестовый Пользователь"
) -> tuple[dict[str, str], str]:
    client.post(
        "/api/v1/auth/register",
        json={"email": email, "password": PASSWORD, "full_name": full_name},
    )
    token_resp = client.post(
        "/api/v1/auth/login", json={"email": email, "password": PASSWORD}
    )
    token = token_resp.json()["access_token"]
    user_id: str = client.get(
        "/api/v1/users/me", headers={"Authorization": f"Bearer {token}"}
    ).json()["id"]
    return {"Authorization": f"Bearer {token}"}, user_id


def test_delete_account_success(api_client: TestClient, run_sql: RunSql) -> None:
    storage = MemoryStorage()
    api_client.app.state.storage = storage  # type: ignore[attr-defined]

    headers_1, user_1_id = register_and_login(
        api_client, "user1@example.md", "Пользователь 1"
    )
    headers_2, user_2_id = register_and_login(
        api_client, "user2@example.md", "Пользователь 2"
    )

    # Создаём участки
    v1_id: str = api_client.post(
        "/api/v1/vineyards",
        headers=headers_1,
        json={"name": "Участок 1", "geometry": SQUARE},
    ).json()["id"]
    v2_id: str = api_client.post(
        "/api/v1/vineyards",
        headers=headers_2,
        json={"name": "Участок 2", "geometry": SQUARE},
    ).json()["id"]

    # Создаём фото и диагнозы
    img1_id = str(uuid.uuid4())
    img1_key = f"{user_1_id}/{img1_id}.jpg"
    diag1_id = str(uuid.uuid4())
    storage.put(img1_key, b"img1", "image/jpeg")
    storage.put(thumb_key(img1_key), b"thumb1", "image/jpeg")
    storage.put(overlay_key(img1_key, "heatmap"), b"heat1", "image/png")
    storage.put(overlay_key(img1_key, "lesion"), b"lesion1", "image/png")

    run_sql(
        f"INSERT INTO leaf_images (id, owner_id, vineyard_id, storage_key, content_type, size_bytes, width, height, latitude, longitude) "
        f"VALUES ('{img1_id}', '{user_1_id}', '{v1_id}', '{img1_key}', 'image/jpeg', 100, 10, 10, 47.14, 28.86)"
    )
    run_sql(
        f"INSERT INTO disease_results (id, image_id, model_name, model_version, status, predicted_class, confidence) "
        f"VALUES ('{diag1_id}', '{img1_id}', 'test_model', '1.0', 'completed', 'downy_mildew', 0.95)"
    )

    # Фото и диагноз для user 2
    img2_id = str(uuid.uuid4())
    img2_key = f"{user_2_id}/{img2_id}.jpg"
    storage.put(img2_key, b"img2", "image/jpeg")
    run_sql(
        f"INSERT INTO leaf_images (id, owner_id, vineyard_id, storage_key, content_type, size_bytes, width, height) "
        f"VALUES ('{img2_id}', '{user_2_id}', '{v2_id}', '{img2_key}', 'image/jpeg', 100, 10, 10)"
    )

    # Проверка лозы для user 1
    vc_id = str(uuid.uuid4())
    vc_key = f"{user_1_id}/vine/{vc_id}.jpg"
    storage.put(vc_key, b"vine_photo", "image/jpeg")
    run_sql(
        f"INSERT INTO vine_checks (id, owner_id, vineyard_id, storage_key, content_type, width, height, model_name, model_version, trunks, bunches, detections) "
        f"VALUES ('{vc_id}', '{user_1_id}', '{v1_id}', '{vc_key}', 'image/jpeg', 10, 10, 'vine_model', '1.0', 5, 20, '[]'::jsonb)"
    )

    # Заметка с фото для user 1
    fn_id = str(uuid.uuid4())
    fn_key = f"{user_1_id}/notes/{fn_id}.jpg"
    storage.put(fn_key, b"note_photo", "image/jpeg")
    run_sql(
        f"INSERT INTO field_notes (id, vineyard_id, noted_on, text, photo_key, photo_type) "
        f"VALUES ('{fn_id}', '{v1_id}', '2026-06-01', 'Заметка', '{fn_key}', 'image/jpeg')"
    )

    assert len(storage.files) == 7

    # Удаляем аккаунт user 1
    delete_resp = api_client.request(
        "DELETE", URL_DELETE, headers=headers_1, json={"password": PASSWORD}
    )
    assert delete_resp.status_code == 204
    # Проверяем удаление refresh cookie
    set_cookie = delete_resp.headers.get("set-cookie", "")
    assert "refresh_token=" in set_cookie

    # Проверяем базу данных
    assert run_sql(f"SELECT id FROM users WHERE id = '{user_1_id}'") == []
    assert run_sql(f"SELECT id FROM vineyards WHERE owner_id = '{user_1_id}'") == []
    assert run_sql(f"SELECT id FROM leaf_images WHERE owner_id = '{user_1_id}'") == []
    assert run_sql(f"SELECT id FROM disease_results WHERE id = '{diag1_id}'") == []
    assert run_sql(f"SELECT id FROM vine_checks WHERE owner_id = '{user_1_id}'") == []
    assert run_sql(f"SELECT id FROM field_notes WHERE id = '{fn_id}'") == []

    # Данные user 2 не тронуты
    assert len(run_sql(f"SELECT id FROM users WHERE id = '{user_2_id}'")) == 1
    assert len(run_sql(f"SELECT id FROM vineyards WHERE owner_id = '{user_2_id}'")) == 1
    assert (
        len(run_sql(f"SELECT id FROM leaf_images WHERE owner_id = '{user_2_id}'")) == 1
    )

    # Файлы user 1 удалены из хранилища, файл user 2 остался
    assert img1_key not in storage.files
    assert thumb_key(img1_key) not in storage.files
    assert overlay_key(img1_key, "heatmap") not in storage.files
    assert overlay_key(img1_key, "lesion") not in storage.files
    assert vc_key not in storage.files
    assert fn_key not in storage.files
    assert img2_key in storage.files

    # Повторный запрос с тем же токеном возвращает 401
    repeat_resp = api_client.get(URL_DELETE, headers=headers_1)
    assert repeat_resp.status_code == 401


def test_delete_account_wrong_password(api_client: TestClient, run_sql: RunSql) -> None:
    headers, user_id = register_and_login(api_client, "wrong_pw@example.md")

    response = api_client.request(
        "DELETE", URL_DELETE, headers=headers, json={"password": "incorrect_password"}
    )
    assert response.status_code == 401
    assert response.json()["detail"] == "Неверный email или пароль"

    # Пользователь на месте
    assert len(run_sql(f"SELECT id FROM users WHERE id = '{user_id}'")) == 1


def test_delete_last_admin_conflict(api_client: TestClient, run_sql: RunSql) -> None:
    headers, user_id = register_and_login(api_client, "admin1@example.md")
    set_role(run_sql, "admin1@example.md", "admin")

    response = api_client.request(
        "DELETE", URL_DELETE, headers=headers, json={"password": PASSWORD}
    )
    assert response.status_code == 409
    assert "последнего администратора" in response.json()["detail"]

    # Администратор не удалён
    assert len(run_sql(f"SELECT id FROM users WHERE id = '{user_id}'")) == 1


def test_delete_admin_when_second_admin_exists(
    api_client: TestClient, run_sql: RunSql
) -> None:
    headers_1, user_1_id = register_and_login(api_client, "adm1@example.md")
    _headers_2, user_2_id = register_and_login(api_client, "adm2@example.md")
    set_role(run_sql, "adm1@example.md", "admin")
    set_role(run_sql, "adm2@example.md", "admin")

    response = api_client.request(
        "DELETE", URL_DELETE, headers=headers_1, json={"password": PASSWORD}
    )
    assert response.status_code == 204

    # Первый удалён, второй остался
    assert run_sql(f"SELECT id FROM users WHERE id = '{user_1_id}'") == []
    assert len(run_sql(f"SELECT id FROM users WHERE id = '{user_2_id}'")) == 1


class FailingDeleteStorage(MemoryStorage):
    def delete(self, key: str) -> None:
        raise RuntimeError("storage disk failure")


def test_delete_account_storage_failure_still_deletes_db(
    api_client: TestClient, run_sql: RunSql
) -> None:
    api_client.app.state.storage = FailingDeleteStorage()  # type: ignore[attr-defined]
    headers, user_id = register_and_login(api_client, "fail_storage@example.md")

    # Добавляем фото
    img_id = str(uuid.uuid4())
    img_key = f"{user_id}/{img_id}.jpg"
    run_sql(
        f"INSERT INTO leaf_images (id, owner_id, storage_key, content_type, size_bytes, width, height) "
        f"VALUES ('{img_id}', '{user_id}', '{img_key}', 'image/jpeg', 100, 10, 10)"
    )

    response = api_client.request(
        "DELETE", URL_DELETE, headers=headers, json={"password": PASSWORD}
    )
    assert response.status_code == 204
    # Запись в БД удалена, несмотря на сбой хранилища
    assert run_sql(f"SELECT id FROM users WHERE id = '{user_id}'") == []


def test_agronomist_review_on_foreign_diagnosis_remains(
    api_client: TestClient, run_sql: RunSql
) -> None:
    headers_grower, grower_id = register_and_login(api_client, "grower@example.md")
    headers_agro, agro_id = register_and_login(api_client, "agro@example.md")
    set_role(run_sql, "agro@example.md", "agronomist")

    v_id: str = api_client.post(
        "/api/v1/vineyards",
        headers=headers_grower,
        json={"name": "Виноградник", "geometry": SQUARE},
    ).json()["id"]

    img_id = str(uuid.uuid4())
    diag_id = str(uuid.uuid4())
    now = datetime.now(UTC).isoformat()
    run_sql(
        f"INSERT INTO leaf_images (id, owner_id, vineyard_id, storage_key, content_type, size_bytes, width, height) "
        f"VALUES ('{img_id}', '{grower_id}', '{v_id}', '{grower_id}/{img_id}.jpg', 'image/jpeg', 100, 10, 10)"
    )
    run_sql(
        f"INSERT INTO disease_results (id, image_id, model_name, model_version, status, predicted_class, confidence, expert_class, expert_comment, reviewed_by_id, reviewed_at) "
        f"VALUES ('{diag_id}', '{img_id}', 'm', '1', 'completed', 'downy_mildew', 0.9, 'downy_mildew', 'Подтверждаю', '{agro_id}', '{now}')"
    )

    # Агроном удаляет свой аккаунт
    delete_resp = api_client.request(
        "DELETE", URL_DELETE, headers=headers_agro, json={"password": PASSWORD}
    )
    assert delete_resp.status_code == 204

    # Проверка виноградаря осталась, reviewed_by_id стал NULL, заключение осталось
    row = run_sql(
        f"SELECT expert_class, expert_comment, reviewed_by_id FROM disease_results WHERE id = '{diag_id}'"
    )
    assert row == [("downy_mildew", "Подтверждаю", None)]


def test_export_account_data(api_client: TestClient, run_sql: RunSql) -> None:
    headers, user_id = register_and_login(
        api_client, "export_user@example.md", "Экспорт Тест"
    )
    headers_other, other_id = register_and_login(api_client, "other_user@example.md")

    # Участок своего пользователя
    v_id: str = api_client.post(
        "/api/v1/vineyards",
        headers=headers,
        json={"name": "Мой участок", "geometry": SQUARE, "planting_year": 2019},
    ).json()["id"]

    # Участок чужого пользователя
    other_v_id: str = api_client.post(
        "/api/v1/vineyards",
        headers=headers_other,
        json={"name": "Чужой участок", "geometry": SQUARE},
    ).json()["id"]

    # Обработка
    api_client.post(
        f"/api/v1/vineyards/{v_id}/treatments",
        headers=headers,
        json={
            "applied_on": "2026-06-01",
            "fungicide": Fungicide.CONTACT.value,
            "product": "Бордоская жидкость",
            "waiting_days": 21,
        },
    )

    # Урожай
    api_client.post(
        f"/api/v1/vineyards/{v_id}/harvests",
        headers=headers,
        json={
            "season": 2026,
            "estimated_kg": 4500.0,
            "actual_kg": 4600.0,
            "sugar_brix": 21.5,
        },
    )

    # Заметки: с координатами и без
    run_sql(
        f"INSERT INTO field_notes (id, vineyard_id, noted_on, text, latitude, longitude, photo_key, photo_type) "
        f"VALUES ('{uuid.uuid4()}', '{v_id}', '2026-06-02', 'Текст заметки', 47.14, 28.86, '{user_id}/notes/1.jpg', 'image/jpeg')"
    )
    run_sql(
        f"INSERT INTO field_notes (id, vineyard_id, noted_on, text) "
        f"VALUES ('{uuid.uuid4()}', '{v_id}', '2026-06-03', 'Заметка без фото')"
    )

    # Диагноз с координатами и заключением агронома
    img_id = str(uuid.uuid4())
    diag_id = str(uuid.uuid4())
    now = datetime.now(UTC).isoformat()
    run_sql(
        f"INSERT INTO leaf_images (id, owner_id, vineyard_id, storage_key, content_type, size_bytes, width, height, latitude, longitude) "
        f"VALUES ('{img_id}', '{user_id}', '{v_id}', '{user_id}/{img_id}.jpg', 'image/jpeg', 100, 10, 10, 47.14, 28.86)"
    )
    run_sql(
        f"INSERT INTO disease_results (id, image_id, model_name, model_version, status, predicted_class, confidence, expert_class, expert_comment, reviewed_by_id, reviewed_at) "
        f"VALUES ('{diag_id}', '{img_id}', 'm', '1', 'completed', 'downy_mildew', 0.92, 'downy_mildew', 'Все верно', '{other_id}', '{now}')"
    )

    # Диагноз чужого пользователя
    other_img_id = str(uuid.uuid4())
    run_sql(
        f"INSERT INTO leaf_images (id, owner_id, vineyard_id, storage_key, content_type, size_bytes, width, height) "
        f"VALUES ('{other_img_id}', '{other_id}', '{other_v_id}', '{other_id}/{other_img_id}.jpg', 'image/jpeg', 100, 10, 10)"
    )
    run_sql(
        f"INSERT INTO disease_results (id, image_id, model_name, model_version, status, predicted_class, confidence) "
        f"VALUES ('{uuid.uuid4()}', '{other_img_id}', 'm', '1', 'completed', 'powdery_mildew', 0.88)"
    )

    # Проверка лозы
    run_sql(
        f"INSERT INTO vine_checks (id, owner_id, vineyard_id, storage_key, content_type, width, height, model_name, model_version, trunks, bunches, detections) "
        f"VALUES ('{uuid.uuid4()}', '{user_id}', '{v_id}', '{user_id}/vine/check.jpg', 'image/jpeg', 10, 10, 'vm', '1', 12, 45, '[]'::jsonb)"
    )

    # Предупреждение
    run_sql(
        f"INSERT INTO alerts (id, user_id, vineyard_id, disease, day, level, read_at) "
        f"VALUES ('{uuid.uuid4()}', '{user_id}', '{v_id}', '{Disease.DOWNY_MILDEW.value}', '2026-06-04', '{RiskLevel.HIGH.value}', '{now}')"
    )
    # Чужое предупреждение
    run_sql(
        f"INSERT INTO alerts (id, user_id, vineyard_id, disease, day, level) "
        f"VALUES ('{uuid.uuid4()}', '{other_id}', '{other_v_id}', '{Disease.POWDERY_MILDEW.value}', '2026-06-04', '{RiskLevel.MEDIUM.value}')"
    )

    resp = api_client.get(URL_EXPORT, headers=headers)
    assert resp.status_code == 200
    today = local_today().isoformat()
    assert (
        resp.headers["content-disposition"]
        == f'attachment; filename="vitiguard-data-{today}.json"'
    )
    assert "application/json" in resp.headers["content-type"]

    data = resp.json()
    assert "exported_at" in data
    assert "Фотографии не включены" in data["notes"]

    # Профиль
    profile = data["profile"]
    assert profile["email"] == "export_user@example.md"
    assert profile["full_name"] == "Экспорт Тест"
    assert profile["role"] == "user"
    assert "created_at" in profile
    assert profile["notification_settings"]["alert_level"] == "high"
    assert profile["telegram_linked"] is False

    # Участки: только свой участок
    assert len(data["vineyards"]) == 1
    vineyard = data["vineyards"][0]
    assert vineyard["id"] == v_id
    assert vineyard["name"] == "Мой участок"
    assert vineyard["planting_year"] == 2019
    assert vineyard["geometry"]["type"] == "MultiPolygon"
    assert vineyard["area_ha"] > 0

    # Журналы участков
    field_logs = data["field_logs"]
    assert len(field_logs["treatments"]) == 1
    assert field_logs["treatments"][0]["product"] == "Бордоская жидкость"
    assert len(field_logs["harvests"]) == 1
    assert field_logs["harvests"][0]["estimated_kg"] == 4500.0
    assert len(field_logs["notes"]) == 2

    # Диагнозы
    assert len(data["diagnoses"]) == 1
    diagnosis = data["diagnoses"][0]
    assert diagnosis["predicted_class"] == "downy_mildew"
    assert diagnosis["coordinates"]["latitude"] == 47.14
    assert diagnosis["agronomist_conclusion"]["disease"] == "downy_mildew"
    assert diagnosis["agronomist_conclusion"]["comment"] == "Все верно"

    # Проверки лозы
    assert len(data["vine_checks"]) == 1
    assert data["vine_checks"][0]["trunks"] == 12
    assert data["vine_checks"][0]["bunches"] == 45

    # Предупреждения
    assert len(data["alerts"]) == 1
    assert data["alerts"][0]["disease"] == "downy_mildew"
    assert data["alerts"][0]["level"] == "high"
    assert data["alerts"][0]["is_read"] is True

    # Проверка отсутствия секретных полей
    raw_text = resp.text
    assert "hashed_password" not in raw_text
    assert "password_hash" not in raw_text
    assert "telegram_chat_id" not in raw_text
    assert "storage_key" not in raw_text
    assert "refresh_token" not in raw_text
    assert other_id not in raw_text
    assert "other_user@example.md" not in raw_text


def test_export_empty_account(api_client: TestClient) -> None:
    headers, _ = register_and_login(api_client, "empty_user@example.md")

    resp = api_client.get(URL_EXPORT, headers=headers)
    assert resp.status_code == 200
    data = resp.json()
    assert data["vineyards"] == []
    assert data["field_logs"]["treatments"] == []
    assert data["field_logs"]["harvests"] == []
    assert data["field_logs"]["notes"] == []
    assert data["diagnoses"] == []
    assert data["vine_checks"] == []
    assert data["alerts"] == []


def test_endpoints_require_auth(api_client: TestClient) -> None:
    assert (
        api_client.request("DELETE", URL_DELETE, json={"password": "any"}).status_code
        == 401
    )
    assert api_client.get(URL_EXPORT).status_code == 401
