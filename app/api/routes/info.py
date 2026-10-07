from fastapi import APIRouter

from app import __version__
from app.api.deps import SettingsDep
from app.schemas.info import AppInfo

router = APIRouter(tags=["service"])


@router.get("/info", summary="Информация о сервисе")
async def get_info(settings: SettingsDep) -> AppInfo:
    return AppInfo(
        name=settings.app_name,
        version=__version__,
        environment=settings.environment,
    )
