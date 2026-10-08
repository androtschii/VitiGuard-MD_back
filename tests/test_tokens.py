import base64
import json
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import jwt
import pytest
from fastapi.testclient import TestClient
from jwt.warnings import InsecureKeyLengthWarning
from pydantic import ValidationError

from app.core.config import DEV_SECRET_KEY, Environment, Settings
from app.core.errors import UnauthorizedError
from app.core.tokens import (
    ISSUER,
    TokenType,
    create_access_token,
    create_refresh_token,
    decode_token,
    hash_token,
)
from app.main import create_app
from app.models.user import UserRole

USER_ID = uuid.UUID("01a115b1-db4b-7ecf-aaa2-9e3f35a32d11")
SETTINGS = Settings(_env_file=None, environment="test")
KEY = SETTINGS.secret_key.get_secret_value()


def forge(claims: dict[str, Any], key: str = KEY, algorithm: str = "HS256") -> str:
    """Токен с произвольным содержимым, подписанный указанным ключом."""
    return jwt.encode(claims, key, algorithm=algorithm)


def valid_claims(**overrides: Any) -> dict[str, Any]:
    now = datetime.now(UTC)
    claims: dict[str, Any] = {
        "iss": ISSUER,
        "sub": str(USER_ID),
        "type": "access",
        "role": "user",
        "iat": now,
        "exp": now + timedelta(minutes=5),
    }
    claims.update(overrides)
    return claims


def test_access_token_carries_user_role_and_lifetime() -> None:
    now = datetime(2026, 10, 1, 12, 0, tzinfo=UTC)
    issued = create_access_token(USER_ID, UserRole.AGRONOMIST, SETTINGS, now=now)

    payload = jwt.decode(
        issued.token, KEY, algorithms=["HS256"], options={"verify_exp": False}
    )

    assert payload["sub"] == str(USER_ID)
    assert payload["role"] == "agronomist"
    assert payload["type"] == "access"
    assert payload["iss"] == ISSUER
    assert issued.expires_at == now + timedelta(minutes=15)
    assert payload["exp"] - payload["iat"] == 15 * 60


def test_access_token_roundtrip() -> None:
    issued = create_access_token(USER_ID, UserRole.ADMIN, SETTINGS)

    claims = decode_token(issued.token, TokenType.ACCESS, SETTINGS)

    assert claims.subject == USER_ID
    assert claims.role is UserRole.ADMIN
    assert claims.token_type is TokenType.ACCESS
    assert claims.token_id == issued.token_id


def test_refresh_token_roundtrip_has_no_role() -> None:
    issued = create_refresh_token(USER_ID, SETTINGS)

    claims = decode_token(issued.token, TokenType.REFRESH, SETTINGS)

    assert claims.subject == USER_ID
    assert claims.role is None
    assert claims.expires_at - claims.issued_at == timedelta(days=30)


def test_every_refresh_token_is_unique() -> None:
    first = create_refresh_token(USER_ID, SETTINGS)
    second = create_refresh_token(USER_ID, SETTINGS)

    assert first.token != second.token
    assert first.token_id != second.token_id
    assert hash_token(first.token) != hash_token(second.token)


def test_ttl_comes_from_settings() -> None:
    settings = Settings(
        _env_file=None,
        environment="test",
        access_token_ttl_minutes=1,
        refresh_token_ttl_days=2,
    )
    now = datetime(2026, 10, 8, tzinfo=UTC)

    assert create_access_token(
        USER_ID, UserRole.USER, settings, now=now
    ).expires_at == now + timedelta(minutes=1)
    assert create_refresh_token(USER_ID, settings, now=now).expires_at == now + (
        timedelta(days=2)
    )


def test_hash_token_is_stable_sha256_hex() -> None:
    assert hash_token("abc") == hash_token("abc")
    assert (
        hash_token("abc")
        == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )


def test_expired_token_is_reported_as_expired() -> None:
    long_ago = datetime.now(UTC) - timedelta(hours=1)
    token = create_access_token(USER_ID, UserRole.USER, SETTINGS, now=long_ago).token

    with pytest.raises(UnauthorizedError, match="Срок действия токена истёк"):
        decode_token(token, TokenType.ACCESS, SETTINGS)


def test_refresh_token_cannot_be_used_as_access_and_back() -> None:
    access = create_access_token(USER_ID, UserRole.USER, SETTINGS).token
    refresh = create_refresh_token(USER_ID, SETTINGS).token

    with pytest.raises(UnauthorizedError, match="Недействительный токен"):
        decode_token(refresh, TokenType.ACCESS, SETTINGS)
    with pytest.raises(UnauthorizedError, match="Недействительный токен"):
        decode_token(access, TokenType.REFRESH, SETTINGS)


def test_token_signed_with_another_key_is_rejected() -> None:
    token = forge(valid_claims(), key="another-key-" + "x" * 40)

    with pytest.raises(UnauthorizedError, match="Недействительный токен"):
        decode_token(token, TokenType.ACCESS, SETTINGS)


def test_tampered_payload_is_rejected() -> None:
    token = create_access_token(USER_ID, UserRole.USER, SETTINGS).token
    header, payload, signature = token.split(".")
    claims = json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))
    claims["role"] = "admin"
    forged_payload = (
        base64.urlsafe_b64encode(json.dumps(claims).encode()).rstrip(b"=").decode()
    )

    with pytest.raises(UnauthorizedError):
        decode_token(
            f"{header}.{forged_payload}.{signature}", TokenType.ACCESS, SETTINGS
        )


def test_token_without_signature_algorithm_none_is_rejected() -> None:
    def encode(part: dict[str, Any]) -> str:
        raw = json.dumps(part, default=str).encode()
        return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()

    claims = valid_claims()
    claims["iat"] = int(claims["iat"].timestamp())
    claims["exp"] = int(claims["exp"].timestamp())
    token = f"{encode({'alg': 'none', 'typ': 'JWT'})}.{encode(claims)}."

    with pytest.raises(UnauthorizedError, match="Недействительный токен"):
        decode_token(token, TokenType.ACCESS, SETTINGS)


def test_token_with_other_algorithm_is_rejected() -> None:
    # PyJWT предупреждает, что ключ короче рекомендованных 64 байт для SHA-512; для
    # проверки отказа по алгоритму важен именно тот же ключ, поэтому ключ не меняем
    with pytest.warns(InsecureKeyLengthWarning):
        token = forge(valid_claims(), algorithm="HS512")

    with pytest.raises(UnauthorizedError, match="Недействительный токен"):
        decode_token(token, TokenType.ACCESS, SETTINGS)


@pytest.mark.parametrize("missing", ["exp", "iat", "sub", "iss", "type"])
def test_token_without_required_claim_is_rejected(missing: str) -> None:
    claims = valid_claims()
    del claims[missing]

    with pytest.raises(UnauthorizedError, match="Недействительный токен"):
        decode_token(forge(claims), TokenType.ACCESS, SETTINGS)


@pytest.mark.parametrize(
    "overrides",
    [
        {"iss": "someone-else"},
        {"sub": "not-a-uuid"},
        {"role": "superuser"},
    ],
)
def test_token_with_wrong_claim_values_is_rejected(overrides: dict[str, Any]) -> None:
    with pytest.raises(UnauthorizedError, match="Недействительный токен"):
        decode_token(forge(valid_claims(**overrides)), TokenType.ACCESS, SETTINGS)


@pytest.mark.parametrize("token", ["", "garbage", "a.b.c", "....."])
def test_garbage_is_rejected(token: str) -> None:
    with pytest.raises(UnauthorizedError, match="Недействительный токен"):
        decode_token(token, TokenType.ACCESS, SETTINGS)


def test_unauthorized_response_names_the_scheme() -> None:
    app = create_app(Settings(_env_file=None, environment="test"))

    @app.get("/secret")
    async def secret() -> None:
        raise UnauthorizedError("Недействительный токен")

    with TestClient(app) as client:
        response = client.get("/secret")

    assert response.status_code == 401
    assert response.headers["WWW-Authenticate"] == "Bearer"
    assert response.json() == {"detail": "Недействительный токен"}


class TestSecretKeyValidation:
    def test_development_key_is_allowed_locally_and_in_tests(self) -> None:
        environments: tuple[Environment, ...] = ("local", "test")
        for environment in environments:
            settings = Settings(_env_file=None, environment=environment)
            assert settings.secret_key.get_secret_value() == DEV_SECRET_KEY

    @pytest.mark.parametrize("environment", ["staging", "production"])
    def test_development_key_is_forbidden_elsewhere(self, environment: str) -> None:
        with pytest.raises(ValidationError, match="нужен собственный secret_key"):
            Settings(_env_file=None, environment=environment)  # type: ignore[arg-type]

    def test_short_key_is_forbidden_in_production(self) -> None:
        with pytest.raises(ValidationError, match="не короче 32"):
            Settings(_env_file=None, environment="production", secret_key="short")  # type: ignore[arg-type]

    def test_own_long_key_is_accepted_in_production(self) -> None:
        settings = Settings(
            _env_file=None,
            environment="production",
            secret_key="k" * 48,  # type: ignore[arg-type]
        )

        assert settings.secret_key.get_secret_value() == "k" * 48

    def test_key_is_not_shown_in_repr(self) -> None:
        assert DEV_SECRET_KEY not in repr(Settings(_env_file=None))

    @pytest.mark.parametrize(
        "variable",
        ["VITIGUARD_ACCESS_TOKEN_TTL_MINUTES", "VITIGUARD_REFRESH_TOKEN_TTL_DAYS"],
    )
    def test_lifetime_must_be_positive(
        self, monkeypatch: pytest.MonkeyPatch, variable: str
    ) -> None:
        monkeypatch.setenv(variable, "0")

        with pytest.raises(ValidationError):
            Settings(_env_file=None)
