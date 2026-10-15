from fastapi import APIRouter, status

from app.api.deps import CurrentUserDep, SessionDep
from app.schemas.errors import ErrorResponse
from app.schemas.region import DistrictStats
from app.services.region import district_stats

router = APIRouter(
    prefix="/regions",
    tags=["regions"],
    responses={status.HTTP_401_UNAUTHORIZED: {"model": ErrorResponse}},
)


@router.get("/stats", summary="Сводная статистика по районам Молдовы")
async def read_district_stats(
    user: CurrentUserDep, session: SessionDep
) -> list[DistrictStats]:
    return await district_stats(session, user.id)
