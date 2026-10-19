from app.core.errors import NotFoundError


class MemoryStorage:
    """Хранилище фото в памяти вместо S3 для тестов API."""

    def __init__(self) -> None:
        self.files: dict[str, tuple[bytes, str]] = {}

    def put(self, key: str, data: bytes, content_type: str) -> None:
        self.files[key] = (data, content_type)

    def get(self, key: str) -> bytes:
        if key not in self.files:
            raise NotFoundError("Фото не найдено")
        return self.files[key][0]

    def delete(self, key: str) -> None:
        self.files.pop(key, None)
