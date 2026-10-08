from typing import Any

import pytest
from pydantic import ValidationError

from app.models.user import UserRole
from app.schemas.auth import RegisterRequest


def make(**overrides: Any) -> RegisterRequest:
    data: dict[str, Any] = {
        "email": "grower@example.md",
        "password": "secret123",
        "full_name": "Андрей Бахов",
    }
    data.update(overrides)
    return RegisterRequest.model_validate(data)


def test_valid_request_defaults_to_user_role() -> None:
    assert make().role is UserRole.USER


def test_agronomist_can_be_chosen() -> None:
    assert make(role="agronomist").role is UserRole.AGRONOMIST


@pytest.mark.parametrize("role", ["admin", "superuser", ""])
def test_admin_and_unknown_roles_cannot_be_chosen(role: str) -> None:
    with pytest.raises(ValidationError):
        make(role=role)


def test_email_is_lowercased_and_trimmed() -> None:
    assert make(email="  Grower@Example.MD ").email == "grower@example.md"


@pytest.mark.parametrize("email", ["", "grower", "grower@", "@example.md", "a b@c.md"])
def test_invalid_email_is_rejected(email: str) -> None:
    with pytest.raises(ValidationError):
        make(email=email)


@pytest.mark.parametrize(
    ("password", "message"),
    [
        ("abc1", "at least 8"),
        ("a1" * 65, "at most 128"),
        ("12345678", "хотя бы одну букву"),
        ("abcdefgh", "хотя бы одну цифру"),
        ("________1", "хотя бы одну букву"),
    ],
)
def test_weak_password_is_rejected(password: str, message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        make(password=password)


@pytest.mark.parametrize(
    "password", ["secret123", "Пароль123", "a1" * 64, "pass word 1"]
)
def test_acceptable_passwords(password: str) -> None:
    assert make(password=password).password == password


def test_password_whitespace_is_kept() -> None:
    assert make(password="  secret123  ").password == "  secret123  "


def test_name_is_trimmed_and_must_not_be_empty() -> None:
    assert make(full_name="  Андрей  ").full_name == "Андрей"
    with pytest.raises(ValidationError):
        make(full_name=" а ")


def test_unknown_fields_are_rejected() -> None:
    with pytest.raises(ValidationError, match="Extra inputs"):
        make(is_admin=True)
