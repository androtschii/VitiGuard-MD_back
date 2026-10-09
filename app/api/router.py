from fastapi import APIRouter

from app.api.routes import auth, info, users, weather

api_router = APIRouter()
api_router.include_router(info.router)
api_router.include_router(auth.router)
api_router.include_router(users.router)
api_router.include_router(weather.router)
