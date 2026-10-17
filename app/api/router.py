from fastapi import APIRouter

from app.api.routes import (
    ai,
    auth,
    info,
    osm,
    regions,
    risks,
    users,
    varieties,
    vineyards,
    weather,
)

api_router = APIRouter()
api_router.include_router(info.router)
api_router.include_router(auth.router)
api_router.include_router(users.router)
api_router.include_router(vineyards.router)
api_router.include_router(varieties.router)
api_router.include_router(osm.router)
api_router.include_router(regions.router)
api_router.include_router(risks.router)
api_router.include_router(weather.router)
api_router.include_router(ai.router)
