from fastapi import APIRouter

from app.api.routes import info

api_router = APIRouter()
api_router.include_router(info.router)
