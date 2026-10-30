from fastapi import APIRouter

from app.api.routes import (
    admin,
    ai,
    alerts,
    auth,
    diagnoses,
    field_notes,
    info,
    osm,
    phenology,
    regions,
    reports,
    risks,
    telegram,
    tiles,
    treatments,
    users,
    varieties,
    vineyards,
    weather,
    ws,
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
api_router.include_router(risks.today_router)
api_router.include_router(treatments.router)
api_router.include_router(field_notes.router)
api_router.include_router(phenology.router)
api_router.include_router(telegram.router)
api_router.include_router(reports.router)
api_router.include_router(admin.router)
api_router.include_router(weather.router)
api_router.include_router(tiles.router)
api_router.include_router(ai.router)
api_router.include_router(alerts.router)
api_router.include_router(diagnoses.router)
api_router.include_router(ws.router)
