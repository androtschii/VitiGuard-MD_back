import io
from typing import Protocol

from minio import Minio
from minio.error import MinioException
from urllib3.exceptions import HTTPError

from app.core.config import Settings
from app.core.errors import ExternalServiceError, NotFoundError

UNAVAILABLE = "Хранилище фото недоступно, попробуйте позже"


class Storage(Protocol):
    """Хранилище файлов: фото листов (pr-039). В тестах — в памяти."""

    def put(self, key: str, data: bytes, content_type: str) -> None: ...

    def get(self, key: str) -> bytes: ...

    def delete(self, key: str) -> None: ...


class ObjectStorage:
    """S3-совместимое хранилище (RustFS в Docker Compose). Клиент синхронный:
    вызовы идут в отдельном потоке и не держат обработку других запросов."""

    def __init__(self, client: Minio, bucket: str) -> None:
        self.client = client
        self.bucket = bucket
        self._bucket_ready = False

    @classmethod
    def from_settings(cls, settings: Settings) -> "ObjectStorage":
        client = Minio(
            settings.s3_endpoint,
            access_key=settings.s3_access_key,
            secret_key=settings.s3_secret_key.get_secret_value(),
            secure=settings.s3_secure,
        )
        return cls(client, settings.s3_bucket)

    def _ensure_bucket(self) -> None:
        # Корзина создаётся при первой записи: отдельного шага развёртывания нет
        if not self._bucket_ready:
            if not self.client.bucket_exists(self.bucket):
                self.client.make_bucket(self.bucket)
            self._bucket_ready = True

    def put(self, key: str, data: bytes, content_type: str) -> None:
        try:
            self._ensure_bucket()
            self.client.put_object(
                self.bucket, key, io.BytesIO(data), len(data), content_type
            )
        except (MinioException, HTTPError) as error:
            raise ExternalServiceError(UNAVAILABLE) from error

    def get(self, key: str) -> bytes:
        try:
            response = self.client.get_object(self.bucket, key)
        except MinioException as error:
            if getattr(error, "code", None) == "NoSuchKey":
                raise NotFoundError("Фото не найдено") from error
            raise ExternalServiceError(UNAVAILABLE) from error
        except HTTPError as error:
            raise ExternalServiceError(UNAVAILABLE) from error
        try:
            return bytes(response.read())
        finally:
            response.close()
            response.release_conn()

    def delete(self, key: str) -> None:
        try:
            self.client.remove_object(self.bucket, key)
        except (MinioException, HTTPError) as error:
            raise ExternalServiceError(UNAVAILABLE) from error
