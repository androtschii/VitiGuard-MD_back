import uuid
from typing import Annotated

from fastapi import APIRouter, Query, Response, UploadFile, status

from app.api.deps import CurrentUserDep, PageParamsDep, SessionDep
from app.core.errors import BadRequestError
from app.repositories.vineyard import VineyardRepository, VineyardView
from app.schemas.errors import ErrorResponse
from app.schemas.pagination import Page
from app.schemas.vineyard import (
    BoundaryImportResult,
    NearbyQuery,
    VineyardCreate,
    VineyardNearby,
    VineyardRead,
)
from app.services.boundary_export import ExportFormat, to_geojson, to_shapefile_zip
from app.services.boundary_files import parse_boundaries
from app.services.vineyard import import_boundaries, validate_geometry

# Контуры хозяйства — килобайты; ограничение защищает от чтения в память
# многомегабайтного файла
MAX_IMPORT_BYTES = 5 * 1024 * 1024
MAX_IMPORT_FEATURES = 500

router = APIRouter(
    prefix="/vineyards",
    tags=["vineyards"],
    responses={status.HTTP_401_UNAUTHORIZED: {"model": ErrorResponse}},
)


def to_read(view: VineyardView) -> VineyardRead:
    # Контур хранится как геометрия PostGIS, а в ответ идут GeoJSON и площадь из запроса
    return VineyardRead.model_validate(
        {
            "id": view.vineyard.id,
            "created_at": view.vineyard.created_at,
            "updated_at": view.vineyard.updated_at,
            "name": view.vineyard.name,
            "geometry": view.geometry,
            "area_ha": view.area_ha,
        }
    )


@router.get("", summary="Мои участки")
async def list_vineyards(
    params: PageParamsDep, user: CurrentUserDep, session: SessionDep
) -> Page[VineyardRead]:
    items, total = await VineyardRepository(session).list_for_owner(user.id, params)
    return Page.of([to_read(item) for item in items], total, params)


def export_response(
    views: list[VineyardView], export_format: ExportFormat, basename: str
) -> Response:
    # attachment: браузер сохраняет файл, а не открывает его во вкладке
    if export_format is ExportFormat.shapefile:
        content = to_shapefile_zip(views, basename)
        media_type, filename = "application/zip", f"{basename}.zip"
    else:
        content = to_geojson(views)
        media_type, filename = "application/geo+json", f"{basename}.geojson"
    return Response(
        content,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get(
    "/export",
    summary="Выгрузка всех моих участков",
    response_class=Response,
    responses={
        status.HTTP_200_OK: {
            "content": {"application/geo+json": {}, "application/zip": {}},
            "description": "GeoJSON или архив Shapefile",
        }
    },
)
async def export_vineyards(
    user: CurrentUserDep,
    session: SessionDep,
    export_format: Annotated[
        ExportFormat, Query(alias="format")
    ] = ExportFormat.geojson,
) -> Response:
    views = await VineyardRepository(session).all_for_owner(user.id)
    return export_response(views, export_format, "vineyards")


# Объявлен раньше /{vineyard_id}: иначе «nearby» разбиралось бы как id участка
@router.get("/nearby", summary="Ближайшие участки к точке")
async def nearby_vineyards(
    query: Annotated[NearbyQuery, Query()], user: CurrentUserDep, session: SessionDep
) -> list[VineyardNearby]:
    found = await VineyardRepository(session).nearest(
        user.id, query.lon, query.lat, query.limit, query.max_distance_km * 1000
    )
    return [
        VineyardNearby.model_validate(
            {**to_read(view).model_dump(), "distance_m": distance}
        )
        for view, distance in found
    ]


@router.get(
    "/{vineyard_id}",
    summary="Участок",
    responses={status.HTTP_404_NOT_FOUND: {"model": ErrorResponse}},
)
async def read_vineyard(
    vineyard_id: uuid.UUID, user: CurrentUserDep, session: SessionDep
) -> VineyardRead:
    return to_read(
        await VineyardRepository(session).get_for_owner(user.id, vineyard_id)
    )


@router.post(
    "",
    status_code=status.HTTP_201_CREATED,
    summary="Создание участка по контуру",
    responses={
        status.HTTP_400_BAD_REQUEST: {"model": ErrorResponse},
        status.HTTP_409_CONFLICT: {"model": ErrorResponse},
    },
)
async def create_vineyard(
    data: VineyardCreate, user: CurrentUserDep, session: SessionDep
) -> VineyardRead:
    geometry = data.geometry.model_dump()
    await validate_geometry(session, user.id, geometry)
    repository = VineyardRepository(session)
    # Владелец — всегда текущий пользователь: поля owner_id в запросе нет
    vineyard = await repository.create_from_geojson(user.id, data.name, geometry)
    await session.commit()
    await session.refresh(vineyard)
    return to_read(await repository.view(vineyard))


@router.post(
    "/import",
    summary="Импорт контуров из GeoJSON или KML",
    responses={status.HTTP_400_BAD_REQUEST: {"model": ErrorResponse}},
)
async def import_vineyards(
    file: UploadFile, user: CurrentUserDep, session: SessionDep
) -> BoundaryImportResult:
    content = await file.read(MAX_IMPORT_BYTES + 1)
    if len(content) > MAX_IMPORT_BYTES:
        raise BadRequestError("Файл больше 5 МБ")
    candidates = parse_boundaries(content)
    if not candidates:
        raise BadRequestError("В файле нет объектов")
    if len(candidates) > MAX_IMPORT_FEATURES:
        raise BadRequestError(
            f"В файле больше {MAX_IMPORT_FEATURES} объектов — разделите его"
        )

    created, skipped = await import_boundaries(session, user.id, candidates)
    await session.commit()
    repository = VineyardRepository(session)
    return BoundaryImportResult.model_validate(
        {
            "created": [to_read(await repository.view(item)) for item in created],
            "skipped": [vars(item) for item in skipped],
        }
    )


@router.get(
    "/{vineyard_id}/export",
    summary="Выгрузка участка",
    response_class=Response,
    responses={
        status.HTTP_200_OK: {
            "content": {"application/geo+json": {}, "application/zip": {}},
            "description": "GeoJSON или архив Shapefile",
        },
        status.HTTP_404_NOT_FOUND: {"model": ErrorResponse},
    },
)
async def export_vineyard(
    vineyard_id: uuid.UUID,
    user: CurrentUserDep,
    session: SessionDep,
    export_format: Annotated[
        ExportFormat, Query(alias="format")
    ] = ExportFormat.geojson,
) -> Response:
    view = await VineyardRepository(session).get_for_owner(user.id, vineyard_id)
    # Имя файла — id: название участка может содержать любые символы
    return export_response([view], export_format, f"vineyard-{vineyard_id}")
