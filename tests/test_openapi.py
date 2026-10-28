"""Документация API не отстаёт от кода (pr-094): новая ручка, схема или раздел
без описания роняют этот тест."""

from typing import Any

import pytest

from app.api.openapi import TAGS
from app.core.config import Settings
from app.main import create_app

# Схемы, которые FastAPI создаёт сам: тела форм и ошибка проверки данных
GENERATED = ("Body_", "HTTPValidationError", "ValidationError")


@pytest.fixture(scope="module")
def schema() -> dict[str, Any]:
    return create_app(Settings(_env_file=None, environment="test")).openapi()


def operations(schema: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    return [
        (f"{method.upper()} {path}", operation)
        for path, item in schema["paths"].items()
        for method, operation in item.items()
    ]


def test_api_has_description(schema: dict[str, Any]) -> None:
    assert "Authorization: Bearer" in schema["info"]["description"]
    assert "Retry-After" in schema["info"]["description"]


def test_every_operation_has_summary_and_known_tag(schema: dict[str, Any]) -> None:
    known = {tag["name"] for tag in TAGS}
    for name, operation in operations(schema):
        assert operation.get("summary"), f"{name}: нет summary"
        assert operation.get("tags"), f"{name}: нет тега"
        assert set(operation["tags"]) <= known, f"{name}: тег без описания"


def test_every_declared_tag_is_used_and_described(schema: dict[str, Any]) -> None:
    used = {tag for _, operation in operations(schema) for tag in operation["tags"]}
    for tag in TAGS:
        assert tag["description"], tag["name"]
        assert tag["name"] in used, f"раздел {tag['name']} без ручек"


def test_secured_operations_document_401(schema: dict[str, Any]) -> None:
    for name, operation in operations(schema):
        if "security" in operation:
            assert "401" in operation["responses"], f"{name}: не описан ответ 401"


def test_every_own_schema_has_description(schema: dict[str, Any]) -> None:
    missing = [
        name
        for name, component in schema["components"]["schemas"].items()
        if not name.startswith(GENERATED) and not component.get("description")
    ]

    assert missing == []
