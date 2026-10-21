"""Клиент Telegram Bot API (pr-082): ответы боту и загрузка присланных фото.

Обновления бот получает через вебхук (POST от Telegram на сервер), поэтому
здесь только исходящие запросы. Токен бота — в адресе запроса, в журналы он не
пишется: ошибки сети превращаются в ExternalServiceError без адреса."""

from typing import Any

import httpx2 as httpx
from pydantic import BaseModel, ValidationError

from app.core.config import Settings
from app.core.errors import ExternalServiceError

UNAVAILABLE = "Telegram недоступен"


class _File(BaseModel):
    file_path: str
    file_size: int | None = None


class _FileResponse(BaseModel):
    ok: bool
    result: _File


class TelegramClient:
    def __init__(self, http: httpx.AsyncClient, settings: Settings) -> None:
        if settings.telegram_bot_token is None:
            raise ValueError("Не задан токен Telegram-бота")
        self._http = http
        self._api = settings.telegram_api_url.rstrip("/")
        self._token = settings.telegram_bot_token.get_secret_value()

    async def send_message(self, chat_id: int, text: str) -> None:
        await self._call("sendMessage", {"chat_id": chat_id, "text": text})

    async def download_file(self, file_id: str, max_bytes: int) -> bytes | None:
        """Файл, присланный боту; None — больше max_bytes."""
        data = await self._call("getFile", {"file_id": file_id})
        try:
            file = _FileResponse.model_validate(data).result
        except ValidationError as error:
            raise ExternalServiceError(UNAVAILABLE) from error
        if file.file_size is not None and file.file_size > max_bytes:
            return None
        try:
            response = await self._http.get(
                f"{self._api}/file/bot{self._token}/{file.file_path}"
            )
            response.raise_for_status()
        except httpx.HTTPError:
            raise ExternalServiceError(UNAVAILABLE) from None
        content = response.content
        return None if len(content) > max_bytes else content

    async def _call(self, method: str, payload: dict[str, Any]) -> Any:
        try:
            response = await self._http.post(
                f"{self._api}/bot{self._token}/{method}", json=payload
            )
            response.raise_for_status()
            return response.json()
        except (httpx.HTTPError, ValueError):
            # Без цепочки исключений: в ней адрес запроса с токеном бота
            raise ExternalServiceError(UNAVAILABLE) from None
