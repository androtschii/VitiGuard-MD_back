import uuid

from pydantic import AwareDatetime, BaseModel, ConfigDict


class RequestSchema(BaseModel):
    """Тело и параметры запроса. Лишние поля — ошибка 422: опечатка в имени поля
    не должна молча превращаться в «поле не передано»."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class ResponseSchema(BaseModel):
    """Ответ API. Собирается и из словаря, и прямо из ORM-объекта."""

    model_config = ConfigDict(from_attributes=True)


class EntityResponse(ResponseSchema):
    """Общие поля записей из таблиц с UUIDPrimaryKeyMixin и TimestampMixin."""

    id: uuid.UUID
    # Время только с часовым поясом: наивное время в ответе — ошибка в коде
    created_at: AwareDatetime
    updated_at: AwareDatetime
