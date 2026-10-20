import json
from typing import Any


class MemoryCache:
    """Кэш в памяти вместо Redis для тестов API: у каждого теста свой."""

    def __init__(self) -> None:
        self.values: dict[str, str] = {}
        self.versions: dict[str, int] = {}

    async def get_json(self, key: str) -> Any | None:
        raw = self.values.get(key)
        return None if raw is None else json.loads(raw)

    async def set_json(self, key: str, value: Any, ttl: int) -> None:
        self.values[key] = json.dumps(value)

    async def version(self, name: str) -> int:
        return self.versions.get(name, 0)

    async def bump(self, name: str) -> None:
        self.versions[name] = self.versions.get(name, 0) + 1
