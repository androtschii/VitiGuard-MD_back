from datetime import UTC, datetime

from app.core.errors import NotFoundError


class MemoryStorage:
    """Хранилище фото в памяти вместо S3 для тестов API."""

    def __init__(self) -> None:
        self.files: dict[str, tuple[bytes, str]] = {}
        self.modified: dict[str, datetime] = {}

    def put(self, key: str, data: bytes, content_type: str) -> None:
        self.files[key] = (data, content_type)
        self.modified[key] = datetime.now(UTC)

    def get(self, key: str) -> bytes:
        if key not in self.files:
            raise NotFoundError("Фото не найдено")
        return self.files[key][0]

    def delete(self, key: str) -> None:
        self.files.pop(key, None)
        self.modified.pop(key, None)

    def purge(self, prefix: str, before: datetime) -> int:
        old = [
            key
            for key, modified in self.modified.items()
            if key.startswith(prefix) and modified < before
        ]
        for key in old:
            self.delete(key)
        return len(old)
