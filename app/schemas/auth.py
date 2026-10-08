import re
from typing import Annotated, Literal

from pydantic import AfterValidator, EmailStr, Field, StringConstraints

from app.schemas.base import RequestSchema, ResponseSchema

MIN_PASSWORD_LENGTH = 8
# Argon2 принимает пароль любой длины, но пароль в мегабайты был бы дешёвым
# способом нагрузить сервер
MAX_PASSWORD_LENGTH = 128


def normalize_email(email: str) -> str:
    # Уникальность email в базе учитывает регистр, поэтому адреса сравниваются
    # только в нижнем регистре
    return email.lower()


def check_password_strength(password: str) -> str:
    # Те же правила, что в форме регистрации; проверка на сервере — защита, а не
    # удобство: запрос можно отправить, минуя форму
    if not re.search(r"[^\W\d_]", password):
        raise ValueError("Добавьте хотя бы одну букву")
    if not re.search(r"\d", password):
        raise ValueError("Добавьте хотя бы одну цифру")
    return password


Email = Annotated[EmailStr, AfterValidator(normalize_email)]

# Пробелы в пароле не обрезаются: они могут быть его частью
Password = Annotated[
    str,
    StringConstraints(
        min_length=MIN_PASSWORD_LENGTH,
        max_length=MAX_PASSWORD_LENGTH,
        strip_whitespace=False,
    ),
    AfterValidator(check_password_strength),
]


class LoginRequest(RequestSchema):
    email: Email
    # При входе правила сложности не проверяются: пароль мог быть задан по
    # прежним правилам, а неверный пароль всё равно не пройдёт проверку хэша
    password: Annotated[
        str,
        StringConstraints(
            min_length=1, max_length=MAX_PASSWORD_LENGTH, strip_whitespace=False
        ),
    ]


class TokenResponse(ResponseSchema):
    """Access-токен. Refresh-токен в тело не попадает: он приходит в HttpOnly-cookie,
    и скрипт на странице не может его прочитать."""

    access_token: str
    token_type: Literal["bearer"] = "bearer"


class RegisterRequest(RequestSchema):
    email: Email
    password: Password
    full_name: str = Field(min_length=2, max_length=255)
    # Роль администратора назначает только администратор, при регистрации её
    # выбрать нельзя
    role: Literal["user", "agronomist"] = "user"
