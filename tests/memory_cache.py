import json
from typing import Any


class MemoryCache:
    """Кэш в памяти вместо Redis для тестов API: у каждого теста свой."""

    def __init__(self) -> None:
        self.values: dict[str, bytes] = {}
        self.versions: dict[str, int] = {}

    async def get_bytes(self, key: str) -> bytes | None:
        return self.values.get(key)

    async def set_bytes(self, key: str, value: bytes, ttl: int) -> None:
        self.values[key] = value

    async def get_json(self, key: str) -> Any | None:
        raw = self.values.get(key)
        return None if raw is None else json.loads(raw)

    async def set_json(self, key: str, value: Any, ttl: int) -> None:
        self.values[key] = json.dumps(value).encode()

    async def pop_json(self, key: str) -> Any | None:
        raw = self.values.pop(key, None)
        return None if raw is None else json.loads(raw)

    async def version(self, name: str) -> int:
        return self.versions.get(name, 0)

    async def bump(self, name: str) -> None:
        self.versions[name] = self.versions.get(name, 0) + 1
