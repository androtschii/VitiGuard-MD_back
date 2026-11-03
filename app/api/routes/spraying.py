"""Маршруты API для дифференцированного внесения СЗР и генерации карт опрыскивания (sprint-28)."""

from fastapi import APIRouter, Response, status

from app.api.deps import CurrentUserDep
from app.schemas.errors import ErrorResponse
from app.schemas.spraying import (
    IsoXmlExportRequest,
    IsoXmlExportResponse,
    VRAOptimizationReportDTO,
    VRAPrescriptionRequest,
    VRAPrescriptionResponse,
)
from app.services.spraying import (
    export_task_data_isoxml,
    generate_vra_prescription,
)

router = APIRouter(
    prefix="/spraying",
    tags=["spraying"],
    responses={status.HTTP_401_UNAUTHORIZED: {"model": ErrorResponse}},
)


@router.post(
    "/generate-vra-map",
    summary="Генерация карты дифференцированного внесения СЗР (VRA)",
    response_model=VRAPrescriptionResponse,
)
async def create_vra_map(
    body: VRAPrescriptionRequest,
    _: CurrentUserDep,
) -> VRAPrescriptionResponse:
    """Рассчитывает нормы расхода рабочей жидкости (л/га) и минутный расход форсунок по секторам."""
    return generate_vra_prescription(body.equipment, body.sectors)


@router.post(
    "/export-isoxml",
    summary="Экспорт технологической карты в стандарт ISO 11783-10 (TaskData XML)",
    response_model=IsoXmlExportResponse,
)
async def export_isoxml(
    body: IsoXmlExportRequest,
    _: CurrentUserDep,
) -> IsoXmlExportResponse:
    """Формирует ISO-XML документ для загрузки в бортовой компьютер опрыскивателя через USB/флешку."""
    xml_string = export_task_data_isoxml(body)
    return IsoXmlExportResponse(
        xml_content=xml_string,
        filename=f"TASKDATA_{body.task_name}.XML",
        zones_count=len(body.sectors),
    )


@router.post(
    "/savings-estimate",
    summary="Расчёт экологического и экономического эффекта VRA",
    response_model=VRAOptimizationReportDTO,
)
async def estimate_savings(
    body: VRAPrescriptionRequest,
    _: CurrentUserDep,
) -> VRAOptimizationReportDTO:
    """Сравнивает нормы сплошного опрыскивания с дифференцированным внесением."""
    prescription = generate_vra_prescription(body.equipment, body.sectors)
    return prescription.summary


@router.get(
    "/isoxml/download-sample",
    summary="Скачивание эталонного XML-шаблона для терминалов опрыскивателя",
)
async def download_sample_isoxml(
    _: CurrentUserDep,
) -> Response:
    """Возвращает базовый пример TaskData XML с заголовками application/xml."""
    sample_xml = (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        '<ISO11783_TaskData VersionMajor="4" VersionMinor="3" '
        'ManagementSoftwareManufacturer="VitiGuard MD" '
        'ManagementSoftwareVersion="2.4.0" DataTransferOrigin="1">\n'
        '  <PFD A="PFD1" C="Demo_Vineyard"/>\n'
        '  <DVC A="DVC1" B="Sprayer_Generic"/>\n'
        '  <TSK A="TSK1" B="VRA_Treatment" E="PFD1" G="1"/>\n'
        "</ISO11783_TaskData>\n"
    )
    return Response(
        content=sample_xml,
        media_type="application/xml",
        headers={"Content-Disposition": 'attachment; filename="TASKDATA.XML"'},
    )
