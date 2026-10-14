from typing import Annotated, Any

from fastapi import APIRouter, Query, status
from pydantic import BaseModel

from app.api.deps import AdminDep, CurrentUserDep, SessionDep
from app.core.errors import BadRequestError
from app.schemas.errors import ErrorResponse
from app.services.osm import osm_vineyards_in_bbox
from app.tasks.osm import import_vineyards
from app.worker.celery_app import celery_app

router = APIRouter(
    prefix="/osm",
    tags=["osm"],
    responses={status.HTTP_401_UNAUTHORIZED: {"model": ErrorResponse}},
)

# Окно карты больше градуса по каждой стороне — это уже масштаб страны, где
# полигоны виноградников мельче пикселя и рисовать их незачем
MAX_BBOX_DEGREES = 1.0


class ImportStarted(BaseModel):
    task_id: str


@router.post(
    "/vineyards/import",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Импорт виноградников Молдовы из OpenStreetMap",
    responses={status.HTTP_403_FORBIDDEN: {"model": ErrorResponse}},
)
async def start_import(_: AdminDep) -> ImportStarted:
    task = celery_app.send_task(import_vineyards.name)
    return ImportStarted(task_id=task.id)


@router.get(
    "/vineyards",
    summary="Виноградники OpenStreetMap в окне карты",
    responses={status.HTTP_400_BAD_REQUEST: {"model": ErrorResponse}},
)
async def list_osm_vineyards(
    bbox: Annotated[str, Query(description="west,south,east,north в градусах")],
    _: CurrentUserDep,
    session: SessionDep,
) -> dict[str, Any]:
    try:
        west, south, east, north = (float(value) for value in bbox.split(","))
    except ValueError as error:
        raise BadRequestError("bbox: четыре числа через запятую") from error
    if not (-180 <= west < east <= 180 and -90 <= south < north <= 90):
        raise BadRequestError("bbox: границы окна заданы неверно")
    if east - west > MAX_BBOX_DEGREES or north - south > MAX_BBOX_DEGREES:
        raise BadRequestError("bbox: окно больше 1° — приблизьте карту")
    return await osm_vineyards_in_bbox(session, west, south, east, north)
