import uuid
from datetime import UTC, date, datetime
from typing import Annotated

from fastapi import APIRouter, Query, Response, status
from fastapi.concurrency import run_in_threadpool

from app.api.deps import CurrentUserDep, SessionDep
from app.core.errors import BadRequestError
from app.repositories.vineyard import VineyardRepository
from app.schemas.errors import ErrorResponse
from app.services.agromodels import LOCAL_TZ
from app.services.data_export import (
    MAX_DAYS,
    DataFormat,
    collect_tables,
    to_csv_zip,
    to_xlsx,
)
from app.services.report import collect_report, render_report
from app.services.risk import local_today, season_year

router = APIRouter(
    prefix="/vineyards",
    tags=["reports"],
    responses={
        status.HTTP_401_UNAUTHORIZED: {"model": ErrorResponse},
        status.HTTP_404_NOT_FOUND: {"model": ErrorResponse},
    },
)


@router.get(
    "/{vineyard_id}/report.pdf",
    summary="PDF-отчёт о состоянии участка",
    response_class=Response,
    responses={status.HTTP_200_OK: {"content": {"application/pdf": {}}}},
)
async def vineyard_report(
    vineyard_id: uuid.UUID, user: CurrentUserDep, session: SessionDep
) -> Response:
    view = await VineyardRepository(session).get_for_owner(user.id, vineyard_id)
    today = local_today()
    data = await collect_report(
        session, view, today, datetime.now(UTC).astimezone(LOCAL_TZ)
    )
    # Вёрстка PDF занимает процессор: в отдельном потоке она не держит остальные запросы
    pdf = await run_in_threadpool(render_report, data)
    return Response(
        pdf,
        media_type="application/pdf",
        headers={
            "Content-Disposition": (
                f'attachment; filename="vitiguard-report-{today.isoformat()}.pdf"'
            )
        },
    )


MEDIA_TYPES = {
    DataFormat.xlsx: (
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    ),
    DataFormat.csv: "application/zip",
}


@router.get(
    "/{vineyard_id}/data",
    summary="История участка таблицами: XLSX или архив CSV",
    response_class=Response,
    responses={
        status.HTTP_200_OK: {"content": {media: {} for media in MEDIA_TYPES.values()}},
        status.HTTP_400_BAD_REQUEST: {"model": ErrorResponse},
    },
)
async def vineyard_data(
    vineyard_id: uuid.UUID,
    user: CurrentUserDep,
    session: SessionDep,
    data_format: Annotated[DataFormat, Query(alias="format")] = DataFormat.xlsx,
    start: Annotated[
        date | None, Query(description="По умолчанию — 1 марта сезона")
    ] = None,
    end: Annotated[date | None, Query(description="По умолчанию — сегодня")] = None,
) -> Response:
    await VineyardRepository(session).get_for_owner(user.id, vineyard_id)
    today = local_today()
    end = end or today
    start = start or date(season_year(end), 3, 1)
    if start > end:
        raise BadRequestError("Начало периода позже конца")
    if (end - start).days > MAX_DAYS:
        raise BadRequestError("Период длиннее трёх лет: выгрузите его частями")
    tables = await collect_tables(session, vineyard_id, start, end)
    render = to_xlsx if data_format is DataFormat.xlsx else to_csv_zip
    content = await run_in_threadpool(render, tables)
    extension = "xlsx" if data_format is DataFormat.xlsx else "zip"
    name = f"vitiguard-data-{start.isoformat()}-{end.isoformat()}.{extension}"
    return Response(
        content,
        media_type=MEDIA_TYPES[data_format],
        headers={"Content-Disposition": f'attachment; filename="{name}"'},
    )
