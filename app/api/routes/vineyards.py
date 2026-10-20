import uuid
from datetime import UTC, date, datetime, time
from typing import Annotated

from fastapi import APIRouter, Query, Response, UploadFile, status

from app.api.deps import CurrentUserDep, PageParamsDep, SessionDep
from app.core.errors import BadRequestError
from app.models.satellite import IndexType
from app.repositories.vineyard import VineyardRepository, VineyardView
from app.schemas.errors import ErrorResponse
from app.schemas.pagination import Page
from app.schemas.vegetation import (
    IndexValueRead,
    NdviMapRead,
    VegetationObservationRead,
    VegetationRefreshStarted,
)
from app.schemas.vineyard import (
    BoundaryImportResult,
    NearbyQuery,
    VineyardContextRead,
    VineyardCreate,
    VineyardNearby,
    VineyardRead,
    VineyardUpdate,
)
from app.services.boundary_export import ExportFormat, to_geojson, to_shapefile_zip
from app.services.boundary_files import parse_boundaries
from app.services.osm import vineyard_context
from app.services.region import district_of
from app.services.vegetation_history import ndvi_image, vegetation_history
from app.services.vineyard import (
    ensure_variety_exists,
    import_boundaries,
    validate_geometry,
)
from app.tasks.satellite import process_vineyard_task
from app.worker.celery_app import celery_app

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
            "variety": view.vineyard.variety,
            "planting_year": view.vineyard.planting_year,
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
    await ensure_variety_exists(session, data.variety_id)
    repository = VineyardRepository(session)
    # Владелец — всегда текущий пользователь: поля owner_id в запросе нет
    vineyard = await repository.create_from_geojson(
        user.id,
        data.name,
        geometry,
        variety_id=data.variety_id,
        planting_year=data.planting_year,
    )
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


@router.patch(
    "/{vineyard_id}",
    summary="Изменение участка",
    responses={
        status.HTTP_400_BAD_REQUEST: {"model": ErrorResponse},
        status.HTTP_404_NOT_FOUND: {"model": ErrorResponse},
        status.HTTP_409_CONFLICT: {"model": ErrorResponse},
    },
)
async def update_vineyard(
    vineyard_id: uuid.UUID,
    data: VineyardUpdate,
    user: CurrentUserDep,
    session: SessionDep,
) -> VineyardRead:
    repository = VineyardRepository(session)
    vineyard = (await repository.get_for_owner(user.id, vineyard_id)).vineyard
    values = data.model_dump(exclude_unset=True)
    geometry = values.pop("geometry", None)
    if geometry is not None:
        # Новый контур проверяется так же, как при создании, кроме наложения на себя
        await validate_geometry(session, user.id, geometry, exclude_id=vineyard.id)
    await ensure_variety_exists(session, values.get("variety_id"))
    await repository.update_with_geojson(vineyard, geometry, **values)
    await session.commit()
    return to_read(await repository.view(vineyard))


@router.delete(
    "/{vineyard_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Удаление участка",
    responses={status.HTTP_404_NOT_FOUND: {"model": ErrorResponse}},
)
async def delete_vineyard(
    vineyard_id: uuid.UUID, user: CurrentUserDep, session: SessionDep
) -> None:
    repository = VineyardRepository(session)
    vineyard = (await repository.get_for_owner(user.id, vineyard_id)).vineyard
    # Снимки, погода и риски участка удаляются каскадом, фото листьев остаются
    await repository.delete(vineyard)
    await session.commit()


@router.get(
    "/{vineyard_id}/context",
    summary="Местность вокруг участка по OpenStreetMap",
    responses={status.HTTP_404_NOT_FOUND: {"model": ErrorResponse}},
)
async def read_vineyard_context(
    vineyard_id: uuid.UUID, user: CurrentUserDep, session: SessionDep
) -> VineyardContextRead:
    view = await VineyardRepository(session).get_for_owner(user.id, vineyard_id)
    context = await vineyard_context(session, view.vineyard)
    district = await district_of(session, view.vineyard)
    return VineyardContextRead.model_validate({**vars(context), "district": district})


# Больше года за раз не обрабатывается: сезон — десятки снимков, а
# многолетний ряд занял бы обработчик задач на часы
MAX_VEGETATION_DAYS = 366


def season_start(today: date) -> date:
    """Начало вегетационного периода — 1 марта текущего года (до марта —
    прошлого): раньше лоза без листьев, и индексы показывают траву и почву."""
    year = today.year if today.month >= 3 else today.year - 1
    return date(year, 3, 1)


@router.post(
    "/{vineyard_id}/vegetation/refresh",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Расчёт NDVI и NDRE по новым снимкам Sentinel-2",
    responses={
        status.HTTP_400_BAD_REQUEST: {"model": ErrorResponse},
        status.HTTP_404_NOT_FOUND: {"model": ErrorResponse},
    },
)
async def refresh_vegetation(
    vineyard_id: uuid.UUID,
    user: CurrentUserDep,
    session: SessionDep,
    start: Annotated[date | None, Query(description="По умолчанию — 1 марта")] = None,
    end: Annotated[date | None, Query(description="По умолчанию — сегодня")] = None,
) -> VegetationRefreshStarted:
    await VineyardRepository(session).get_for_owner(user.id, vineyard_id)
    today = datetime.now(UTC).date()
    first = start or season_start(today)
    last = end or today
    if first > last:
        raise BadRequestError("Начало периода должно быть не позже конца")
    if (last - first).days > MAX_VEGETATION_DAYS:
        raise BadRequestError("Период — не больше года")
    task = celery_app.send_task(
        process_vineyard_task.name,
        args=[
            str(vineyard_id),
            datetime.combine(first, time.min, UTC).isoformat(),
            datetime.combine(last, time.max, UTC).isoformat(),
        ],
    )
    return VegetationRefreshStarted(task_id=task.id)


@router.get(
    "/{vineyard_id}/vegetation",
    summary="История NDVI и NDRE участка",
    responses={status.HTTP_404_NOT_FOUND: {"model": ErrorResponse}},
)
async def read_vegetation(
    vineyard_id: uuid.UUID, user: CurrentUserDep, session: SessionDep
) -> list[VegetationObservationRead]:
    await VineyardRepository(session).get_for_owner(user.id, vineyard_id)
    observations = []
    for scan in await vegetation_history(session, vineyard_id):
        values = {
            index.index_type: IndexValueRead(
                mean=index.mean_value,
                min=index.min_value,
                max=index.max_value,
                std=index.std_value,
            )
            for index in scan.indices
        }
        observations.append(
            VegetationObservationRead(
                id=scan.id,
                acquired_at=scan.acquired_at,
                platform=scan.platform,
                scene_cloud_cover=scan.cloud_cover,
                parcel_cloud_cover=scan.parcel_cloud_cover,
                ndvi=values.get(IndexType.NDVI),
                ndre=values.get(IndexType.NDRE),
                ndvi_map=(
                    NdviMapRead(corners=scan.ndvi_corners)
                    if scan.ndvi_corners
                    else None
                ),
            )
        )
    return observations


@router.get(
    "/{vineyard_id}/vegetation/{scan_id}/ndvi.png",
    summary="Картограмма NDVI участка по снимку",
    response_class=Response,
    responses={
        status.HTTP_200_OK: {"content": {"image/png": {}}},
        status.HTTP_404_NOT_FOUND: {"model": ErrorResponse},
    },
)
async def read_ndvi_map(
    vineyard_id: uuid.UUID,
    scan_id: uuid.UUID,
    user: CurrentUserDep,
    session: SessionDep,
) -> Response:
    await VineyardRepository(session).get_for_owner(user.id, vineyard_id)
    return Response(
        await ndvi_image(session, vineyard_id, scan_id),
        media_type="image/png",
        headers={"Cache-Control": "private, max-age=86400"},
    )
