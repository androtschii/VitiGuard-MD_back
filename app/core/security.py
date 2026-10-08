import asyncio
from functools import cache

from pwdlib import PasswordHash
from pwdlib.exceptions import UnknownHashError
from pwdlib.hashers.argon2 import Argon2Hasher

# Argon2id с параметрами библиотеки по умолчанию (64 МиБ памяти, 3 прохода,
# 4 потока — профиль из RFC 9106 для серверов): перебор на видеокартах
# дорог из-за требований к памяти. Соль случайная и хранится внутри хэша
_password_hash = PasswordHash((Argon2Hasher(),))


def hash_password(password: str) -> str:
    return _password_hash.hash(password)


@cache
def _dummy_hash() -> str:
    return hash_password("dummy-password-for-timing")


def verify_password(password: str, hashed: str | None) -> bool:
    """Проверяет пароль. Если хэша нет (пользователя с таким email не нашли),
    всё равно выполняет полную проверку на заглушке: иначе ответ «нет такого
    пользователя» приходит на 150 мс быстрее, и по времени ответа можно
    узнавать, какие email зарегистрированы."""
    valid, _ = verify_and_upgrade(password, hashed)
    return valid


def verify_and_upgrade(password: str, hashed: str | None) -> tuple[bool, str | None]:
    """Возвращает (пароль верный, новый хэш). Новый хэш приходит, когда пароль
    верный, а сохранённый хэш сделан со старыми параметрами: его нужно записать
    вместо старого, тогда стоимость хэширования растёт без сброса паролей."""
    if hashed is None:
        _password_hash.verify(password, _dummy_hash())
        return False, None
    try:
        return _password_hash.verify_and_update(password, hashed)
    except UnknownHashError:
        # В базе лежит не распознанная строка: вход невозможен, а не ошибка сервера
        return False, None


# Хэширование занимает ~150 мс чистого процессорного времени. Выполненное прямо в
# обработчике, оно блокировало бы цикл событий: все остальные запросы ждали бы.
# Поэтому из асинхронного кода пароли обрабатываются в потоке
async def ahash_password(password: str) -> str:
    return await asyncio.to_thread(hash_password, password)


async def averify_and_upgrade(
    password: str, hashed: str | None
) -> tuple[bool, str | None]:
    return await asyncio.to_thread(verify_and_upgrade, password, hashed)
