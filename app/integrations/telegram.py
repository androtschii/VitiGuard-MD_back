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

    async def send_message(
        self, chat_id: int, text: str, reply_markup: dict[str, Any] | None = None
    ) -> None:
        payload: dict[str, Any] = {"chat_id": chat_id, "text": text}
        if reply_markup is not None:
            payload["reply_markup"] = reply_markup
        await self._call("sendMessage", payload)

    async def set_commands(self, commands: list[tuple[str, str]]) -> None:
        """Меню команд у поля ввода («/» и кнопка «Меню»)."""
        await self._call(
            "setMyCommands",
            {
                "commands": [
                    {"command": name, "description": description}
                    for name, description in commands
                ]
            },
        )

    async def set_menu_button(self, text: str | None, url: str | None) -> None:
        """Кнопка меню: открывает мини-приложение или, без адреса, список команд."""
        button: dict[str, Any] = (
            {"type": "web_app", "text": text, "web_app": {"url": url}}
            if text and url
            else {"type": "commands"}
        )
        await self._call("setChatMenuButton", {"menu_button": button})

    async def answer_callback(self, callback_id: str, text: str) -> None:
        """Ответ на нажатие кнопки: короткая подсказка поверх чата."""
        await self._call(
            "answerCallbackQuery", {"callback_query_id": callback_id, "text": text}
        )

    async def edit_reply_markup(
        self, chat_id: int, message_id: int, reply_markup: dict[str, Any]
    ) -> None:
        await self._call(
            "editMessageReplyMarkup",
            {
                "chat_id": chat_id,
                "message_id": message_id,
                "reply_markup": reply_markup,
            },
        )

    async def delete_webhook(self) -> None:
        await self._call("deleteWebhook", {})

    async def get_updates(self, offset: int, wait: int) -> list[dict[str, Any]]:
        """Обновления с номера offset; Telegram держит запрос до wait секунд,
        пока не появится новое (long polling)."""
        data = await self._call(
            "getUpdates",
            {
                "offset": offset,
                "timeout": wait,
                "allowed_updates": ["message", "callback_query"],
            },
            wait=wait + 10,
        )
        result = data.get("result") if isinstance(data, dict) else None
        if not isinstance(result, list):
            raise ExternalServiceError(UNAVAILABLE)
        return result

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

    async def _call(
        self, method: str, payload: dict[str, Any], wait: float | None = None
    ) -> Any:
        try:
            response = await self._http.post(
                f"{self._api}/bot{self._token}/{method}",
                json=payload,
                timeout=httpx.USE_CLIENT_DEFAULT if wait is None else wait,
            )
            response.raise_for_status()
            return response.json()
        except (httpx.HTTPError, ValueError):
            # Без цепочки исключений: в ней адрес запроса с токеном бота
            raise ExternalServiceError(UNAVAILABLE) from None
