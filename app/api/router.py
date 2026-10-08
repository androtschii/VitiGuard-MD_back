from fastapi import APIRouter

from app.api.routes import auth, info

api_router = APIRouter()
api_router.include_router(info.router)
api_router.include_router(auth.router)
