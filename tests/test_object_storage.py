from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from minio.error import MinioException
from urllib3.exceptions import HTTPError

from app.core.config import Settings
from app.core.errors import ExternalServiceError, NotFoundError
from app.integrations.object_storage import ObjectStorage


class MissingKey(MinioException):
    code = "NoSuchKey"


class Denied(MinioException):
    code = "AccessDenied"


def storage(exists: bool = True) -> tuple[ObjectStorage, MagicMock]:
    client = MagicMock()
    client.bucket_exists.return_value = exists
    return ObjectStorage(client, "leaf-images"), client


def test_from_settings_uses_endpoint_and_bucket() -> None:
    settings = Settings(_env_file=None, s3_endpoint="storage:9000", s3_bucket="b")

    created = ObjectStorage.from_settings(settings)

    assert created.bucket == "b"
    assert created.client._base_url.host == "storage:9000"


def test_put_creates_bucket_once() -> None:
    store, client = storage(exists=False)

    store.put("u/1.jpg", b"jpeg", "image/jpeg")
    store.put("u/2.jpg", b"jpeg", "image/jpeg")

    client.make_bucket.assert_called_once_with("leaf-images")
    assert client.bucket_exists.call_count == 1
    bucket, key, _, length, content_type = client.put_object.call_args.args
    assert (bucket, key, length, content_type) == (
        "leaf-images",
        "u/2.jpg",
        4,
        "image/jpeg",
    )


def test_get_reads_and_releases_connection() -> None:
    store, client = storage()
    response = client.get_object.return_value
    response.read.return_value = b"jpeg"

    assert store.get("u/1.jpg") == b"jpeg"
    response.close.assert_called_once()
    response.release_conn.assert_called_once()


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (MissingKey(), NotFoundError),
        (Denied(), ExternalServiceError),
        (HTTPError(), ExternalServiceError),
    ],
)
def test_get_errors(error: Exception, expected: type[Exception]) -> None:
    store, client = storage()
    client.get_object.side_effect = error

    with pytest.raises(expected):
        store.get("u/1.jpg")


def test_unavailable_storage_on_put_and_delete() -> None:
    store, client = storage()
    client.put_object.side_effect = HTTPError()
    client.remove_object.side_effect = Denied()

    with pytest.raises(ExternalServiceError):
        store.put("u/1.jpg", b"jpeg", "image/jpeg")
    with pytest.raises(ExternalServiceError):
        store.delete("u/1.jpg")


def test_delete_removes_object() -> None:
    store, client = storage()

    store.delete("u/1.jpg")

    client.remove_object.assert_called_once_with("leaf-images", "u/1.jpg")


def test_purge_removes_old_files_under_prefix() -> None:
    store, client = storage()
    now = datetime(2026, 10, 29, tzinfo=UTC)
    client.list_objects.return_value = [
        SimpleNamespace(
            object_name="jobs/u/old.jpg", last_modified=now - timedelta(days=2)
        ),
        SimpleNamespace(object_name="jobs/u/new.jpg", last_modified=now),
        SimpleNamespace(object_name="jobs/u/", last_modified=None),
    ]

    assert store.purge("jobs/", now - timedelta(days=1)) == 1

    client.list_objects.assert_called_once_with(
        "leaf-images", prefix="jobs/", recursive=True
    )
    client.remove_object.assert_called_once_with("leaf-images", "jobs/u/old.jpg")


def test_purge_with_unavailable_storage() -> None:
    store, client = storage()
    client.list_objects.side_effect = Denied()

    with pytest.raises(ExternalServiceError):
        store.purge("jobs/", datetime(2026, 10, 29, tzinfo=UTC))
