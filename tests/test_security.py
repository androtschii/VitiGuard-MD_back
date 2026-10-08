import threading

import pytest
from pwdlib import PasswordHash
from pwdlib.hashers.argon2 import Argon2Hasher

from app.core import security
from app.core.security import (
    ahash_password,
    averify_and_upgrade,
    hash_password,
    verify_and_upgrade,
    verify_password,
)

PASSWORD = "secret123"


def test_hash_is_argon2id_with_recommended_parameters() -> None:
    hashed = hash_password(PASSWORD)

    assert hashed.startswith("$argon2id$v=19$m=65536,t=3,p=4$")


def test_hash_does_not_contain_password() -> None:
    assert PASSWORD not in hash_password(PASSWORD)


def test_same_password_gets_different_hashes() -> None:
    # Случайная соль: по совпадению хэшей нельзя найти пользователей с одним паролем
    assert hash_password(PASSWORD) != hash_password(PASSWORD)


def test_correct_password_is_accepted_and_wrong_one_rejected() -> None:
    hashed = hash_password(PASSWORD)

    assert verify_password(PASSWORD, hashed) is True
    assert verify_password("secret124", hashed) is False
    assert verify_password("", hashed) is False


def test_passwords_in_any_alphabet() -> None:
    hashed = hash_password("Пароль123")

    assert verify_password("Пароль123", hashed) is True
    assert verify_password("пароль123", hashed) is False


def test_unrecognised_hash_means_failed_login_not_error() -> None:
    assert verify_password(PASSWORD, "plain-text-password") is False
    assert verify_password(PASSWORD, "") is False


def test_missing_hash_is_rejected_but_still_checked_against_dummy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checked: list[str] = []
    original = security._password_hash.verify

    def spy(password: str | bytes, hashed: str | bytes) -> bool:
        checked.append(str(hashed))
        return original(password, hashed)

    monkeypatch.setattr(security._password_hash, "verify", spy)

    assert verify_password(PASSWORD, None) is False
    assert len(checked) == 1
    assert checked[0].startswith("$argon2id$")


def test_dummy_hash_never_accepts_a_password() -> None:
    assert verify_password("dummy-password-for-timing", None) is False


def test_current_hash_does_not_need_upgrade() -> None:
    assert verify_and_upgrade(PASSWORD, hash_password(PASSWORD)) == (True, None)


def test_weaker_hash_is_upgraded_after_successful_check() -> None:
    weak = PasswordHash(
        (Argon2Hasher(time_cost=1, memory_cost=8, parallelism=1),)
    ).hash(PASSWORD)

    valid, upgraded = verify_and_upgrade(PASSWORD, weak)

    assert valid is True
    assert upgraded is not None
    assert upgraded.startswith("$argon2id$v=19$m=65536,t=3,p=4$")
    assert verify_password(PASSWORD, upgraded) is True


def test_weak_hash_is_not_upgraded_for_wrong_password() -> None:
    weak = PasswordHash(
        (Argon2Hasher(time_cost=1, memory_cost=8, parallelism=1),)
    ).hash(PASSWORD)

    assert verify_and_upgrade("secret124", weak) == (False, None)


async def test_async_wrappers_work_in_a_worker_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    threads: list[threading.Thread] = []
    original = security.hash_password

    def spy(password: str) -> str:
        threads.append(threading.current_thread())
        return original(password)

    monkeypatch.setattr(security, "hash_password", spy)

    hashed = await ahash_password(PASSWORD)

    assert threads[0] is not threading.main_thread()
    assert await averify_and_upgrade(PASSWORD, hashed) == (True, None)
    assert await averify_and_upgrade("secret124", hashed) == (False, None)
