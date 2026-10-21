import uuid
from datetime import UTC, datetime

from fastapi import APIRouter, Response, status
from fastapi.concurrency import run_in_threadpool

from app.api.deps import CurrentUserDep, SessionDep
from app.repositories.vineyard import VineyardRepository
from app.schemas.errors import ErrorResponse
from app.services.agromodels import LOCAL_TZ
from app.services.report import collect_report, render_report
from app.services.risk import local_today

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
