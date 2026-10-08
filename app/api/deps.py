from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import Depends, Query, Request
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import Settings
from app.schemas.pagination import PageParams


def get_app_settings(request: Request) -> Settings:
    settings: Settings = request.app.state.settings
    return settings


SettingsDep = Annotated[Settings, Depends(get_app_settings)]


async def get_session(request: Request) -> AsyncIterator[AsyncSession]:
    """Сессия на один запрос. Изменения не сохраняются сами: commit вызывает
    код, который их сделал. При ошибке и в конце запроса всё несохранённое
    откатывается, а соединение возвращается в пул."""
    session_factory: async_sessionmaker[AsyncSession] = (
        request.app.state.session_factory
    )
    async with session_factory() as session:
        yield session


SessionDep = Annotated[AsyncSession, Depends(get_session)]

# Параметры страницы берутся из строки запроса: ?page=2&size=20
PageParamsDep = Annotated[PageParams, Query()]
