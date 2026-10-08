import uuid
from datetime import UTC, datetime

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.api.deps import PageParamsDep
from app.models import User
from app.schemas.base import EntityResponse, RequestSchema
from app.schemas.errors import ErrorResponse
from app.schemas.pagination import MAX_PAGE_SIZE, Page, PageParams


class CreateNote(RequestSchema):
    title: str


class UserBrief(EntityResponse):
    email: str


NOW = datetime(2026, 10, 8, 9, 30, tzinfo=UTC)


def test_request_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        CreateNote.model_validate({"title": "a", "titel": "b"})


def test_request_strips_whitespace() -> None:
    assert CreateNote(title="  Крикова  ").title == "Крикова"


def test_response_is_built_from_orm_object() -> None:
    user = User(
        id=uuid.uuid4(),
        email="grower@example.md",
        hashed_password="hash",
        created_at=NOW,
        updated_at=NOW,
    )

    brief = UserBrief.model_validate(user)

    assert brief.id == user.id
    assert brief.email == "grower@example.md"
    assert "hashed_password" not in brief.model_dump()


def test_response_rejects_naive_datetime() -> None:
    with pytest.raises(ValidationError, match="timezone"):
        UserBrief(
            id=uuid.uuid4(),
            email="a@b.md",
            created_at=datetime(2026, 10, 8, 9, 30),
            updated_at=NOW,
        )


@pytest.mark.parametrize(
    ("total", "size", "pages"),
    [(0, 20, 0), (1, 20, 1), (20, 20, 1), (21, 20, 2), (41, 20, 3)],
)
def test_page_counts_pages(total: int, size: int, pages: int) -> None:
    page = Page[str](items=[], total=total, page=1, size=size)

    assert page.pages == pages


def test_page_is_built_from_params() -> None:
    params = PageParams(page=3, size=10)

    page = Page[str].of(["a", "b"], total=22, params=params)

    assert params.offset == 20
    assert page.model_dump() == {
        "items": ["a", "b"],
        "total": 22,
        "page": 3,
        "size": 10,
        "pages": 3,
    }


def test_error_response_shape() -> None:
    assert ErrorResponse(detail="Участок не найден").model_dump() == {
        "detail": "Участок не найден"
    }


@pytest.fixture
def paged_client() -> TestClient:
    app = FastAPI()

    @app.get("/items")
    async def items(params: PageParamsDep) -> Page[str]:
        return Page[str].of(["a", "b"], total=45, params=params)

    return TestClient(app)


def test_page_params_defaults_come_from_query(paged_client: TestClient) -> None:
    body = paged_client.get("/items").json()

    assert (body["page"], body["size"], body["pages"]) == (1, 20, 3)


def test_page_params_are_read_from_query(paged_client: TestClient) -> None:
    body = paged_client.get("/items", params={"page": 2, "size": 5}).json()

    assert (body["page"], body["size"], body["pages"]) == (2, 5, 9)


@pytest.mark.parametrize(
    "query",
    [
        {"page": 0},
        {"size": 0},
        {"size": MAX_PAGE_SIZE + 1},
        {"page": "abc"},
        {"colour": "red"},
    ],
)
def test_invalid_page_params_rejected(
    paged_client: TestClient, query: dict[str, str | int]
) -> None:
    assert paged_client.get("/items", params=query).status_code == 422


def test_page_appears_in_openapi_with_pages_field(paged_client: TestClient) -> None:
    schema = paged_client.get("/openapi.json").json()["components"]["schemas"]
    page = schema["Page_str_"]

    assert "pages" in page["required"]
    assert page["properties"]["items"]["items"] == {"type": "string"}
