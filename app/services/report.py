"""PDF-отчёт о состоянии виноградника (pr-086): всё, что система знает об
участке, на нескольких страницах — для агронома, хозяина или проверяющего,
которым удобнее бумага или файл, чем кабинет.

Данные собираются из базы (риск, план обработок, спутниковые индексы,
аномалии, проверки листьев), а не запрашиваются у внешних сервисов: отчёт
строится быстро и показывает ровно то, что видно в кабинете."""

import io
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

from reportlab.graphics.charts.lineplots import LinePlot
from reportlab.graphics.shapes import Drawing, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Flowable,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.diagnosis import DiseaseResult, LeafImage
from app.models.enums import Disease, Fungicide, RiskLevel
from app.models.satellite import IndexType
from app.models.treatment import Treatment
from app.models.variety import Variety
from app.models.weather import DiseaseRisk
from app.repositories.vineyard import VineyardView
from app.services.anomaly import Anomaly, vineyard_anomalies
from app.services.risk import LEVEL_ORDER, season_year
from app.services.treatments import Action, Plan, vineyard_plan
from app.services.vegetation_history import vegetation_history

FONTS = Path(__file__).resolve().parent.parent / "assets" / "fonts"
# Встроенные шрифты PDF не знают кириллицы: DejaVu Sans встраивается в файл
FONT, FONT_BOLD = "DejaVuSans", "DejaVuSans-Bold"
RISK_DAYS = 7
RECENT = 10

DISEASES = {Disease.DOWNY_MILDEW: "Милдью", Disease.POWDERY_MILDEW: "Оидиум"}
LEVELS = {
    RiskLevel.LOW: "низкий",
    RiskLevel.MEDIUM: "средний",
    RiskLevel.HIGH: "высокий",
}
LEVEL_COLORS = {
    RiskLevel.LOW: colors.HexColor("#dcfce7"),
    RiskLevel.MEDIUM: colors.HexColor("#fef3c7"),
    RiskLevel.HIGH: colors.HexColor("#fee2e2"),
}
FUNGICIDES = {
    Fungicide.CONTACT: "контактный",
    Fungicide.SYSTEMIC: "системный",
    Fungicide.SULFUR: "сера",
}
ACTIONS = {
    Action.PROTECT: "Защитная обработка",
    Action.CURE: "Лечебная обработка",
    Action.INSPECT: "Осмотр листьев",
}
LEAF_CLASSES = {
    "healthy": "здоровый лист",
    "black_rot": "чёрная гниль",
    "esca": "эска",
    "leaf_blight": "пятнистость листьев",
    "downy_mildew": "милдью",
    "powdery_mildew": "оидиум",
}


@dataclass(frozen=True)
class RiskDay:
    day: date
    downy: RiskLevel | None
    powdery: RiskLevel | None


@dataclass(frozen=True)
class IndexPoint:
    day: date
    ndvi: float | None
    ndre: float | None
    cloud: float | None


@dataclass(frozen=True)
class LeafCheck:
    day: date
    verdict: str
    confidence: float | None
    expert: str | None


@dataclass(frozen=True)
class ReportData:
    name: str
    area_ha: float
    variety: str | None
    planting_year: int | None
    today: date
    created_at: datetime
    risk: list[RiskDay]
    plan: Plan
    treatments: list[Treatment]
    indices: list[IndexPoint]
    anomalies: list[Anomaly]
    checks: list[LeafCheck]


async def collect_report(
    session: AsyncSession, view: VineyardView, today: date, created_at: datetime
) -> ReportData:
    vineyard = view.vineyard
    risks = await session.scalars(
        select(DiseaseRisk).where(
            DiseaseRisk.vineyard_id == vineyard.id,
            DiseaseRisk.target_date >= today,
            DiseaseRisk.target_date <= today + timedelta(days=RISK_DAYS),
        )
    )
    # По болезни — худший уровень из её моделей, как в кабинете
    worst: dict[tuple[date, Disease], RiskLevel] = {}
    for risk in risks:
        key = (risk.target_date, risk.disease)
        if key not in worst or LEVEL_ORDER[risk.level] > LEVEL_ORDER[worst[key]]:
            worst[key] = risk.level
    risk_days = [
        RiskDay(
            day,
            worst.get((day, Disease.DOWNY_MILDEW)),
            worst.get((day, Disease.POWDERY_MILDEW)),
        )
        for day in (today + timedelta(days=i) for i in range(RISK_DAYS + 1))
    ]

    treatments = await session.scalars(
        select(Treatment)
        .where(Treatment.vineyard_id == vineyard.id)
        .order_by(Treatment.applied_on.desc())
        .limit(RECENT)
    )

    season_start = date(season_year(today), 3, 1)
    indices = []
    for scan in await vegetation_history(session, vineyard.id):
        if scan.acquired_at.date() < season_start:
            continue
        values = {index.index_type: index.mean_value for index in scan.indices}
        indices.append(
            IndexPoint(
                scan.acquired_at.date(),
                values.get(IndexType.NDVI),
                values.get(IndexType.NDRE),
                scan.parcel_cloud_cover,
            )
        )

    rows = await session.execute(
        select(LeafImage.created_at, DiseaseResult)
        .join(DiseaseResult, DiseaseResult.image_id == LeafImage.id)
        .where(LeafImage.vineyard_id == vineyard.id)
        .order_by(LeafImage.created_at.desc())
        .limit(RECENT)
    )
    checks = [
        LeafCheck(
            created.date(),
            LEAF_CLASSES.get(result.predicted_class or "", "—"),
            result.confidence,
            LEAF_CLASSES.get(result.expert_class) if result.expert_class else None,
        )
        for created, result in rows
    ]

    variety = (
        await session.get(Variety, vineyard.variety_id)
        if vineyard.variety_id is not None
        else None
    )
    return ReportData(
        name=vineyard.name,
        area_ha=view.area_ha,
        variety=variety.name if variety else None,
        planting_year=vineyard.planting_year,
        today=today,
        created_at=created_at,
        risk=risk_days,
        plan=await vineyard_plan(session, vineyard.id, today),
        treatments=list(treatments),
        indices=indices,
        anomalies=await vineyard_anomalies(session, vineyard),
        checks=checks,
    )


def _register_fonts() -> None:
    if FONT not in pdfmetrics.getRegisteredFontNames():
        pdfmetrics.registerFont(TTFont(FONT, FONTS / "DejaVuSans.ttf"))
        pdfmetrics.registerFont(TTFont(FONT_BOLD, FONTS / "DejaVuSans-Bold.ttf"))


TEXT = ParagraphStyle(
    "text", fontName=FONT, fontSize=9.5, leading=13, alignment=TA_LEFT
)
SMALL = ParagraphStyle(
    "small", parent=TEXT, fontSize=8, leading=10.5, textColor=colors.grey
)
TITLE = ParagraphStyle(
    "title", parent=TEXT, fontName=FONT_BOLD, fontSize=17, leading=21, spaceAfter=3
)
HEADING = ParagraphStyle(
    "heading",
    parent=TEXT,
    fontName=FONT_BOLD,
    fontSize=12,
    leading=16,
    spaceBefore=10,
    spaceAfter=4,
)


def _date(value: date) -> str:
    return value.strftime("%d.%m.%Y")


def _number(value: float) -> str:
    # Десятичная запятая, как принято в русском тексте
    return f"{value:.2f}".replace(".", ",")


def _table(rows: list[list[str]], widths: list[float]) -> Table:
    table = Table(rows, colWidths=widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("FONT", (0, 0), (-1, -1), FONT, 8.5),
                ("FONT", (0, 0), (-1, 0), FONT_BOLD, 8.5),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#ede9fe")),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#d4d4d8")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    return table


def _risk_table(days: Sequence[RiskDay]) -> Table:
    rows = [["Дата", "Милдью", "Оидиум"]]
    style = []
    for row, item in enumerate(days, start=1):
        rows.append(
            [
                _date(item.day),
                LEVELS[item.downy] if item.downy else "вне сезона",
                LEVELS[item.powdery] if item.powdery else "вне сезона",
            ]
        )
        for column, level in ((1, item.downy), (2, item.powdery)):
            if level is not None:
                style.append(
                    ("BACKGROUND", (column, row), (column, row), LEVEL_COLORS[level])
                )
    table = _table(rows, [35 * mm, 40 * mm, 40 * mm])
    table.setStyle(TableStyle(style))
    return table


def _ndvi_chart(points: Sequence[IndexPoint]) -> Drawing | None:
    clear = [(p.day, p.ndvi) for p in points if p.ndvi is not None]
    if len(clear) < 2:
        return None
    first = clear[0][0]
    drawing = Drawing(170 * mm, 58 * mm)
    plot = LinePlot()
    # LinePlot ждёт координаты в целых пунктах
    plot.x, plot.y = round(12 * mm), round(8 * mm)
    plot.width, plot.height = round(150 * mm), round(42 * mm)
    plot.data = [[((day - first).days, value) for day, value in clear]]
    plot.lines[0].strokeColor = colors.HexColor("#16a34a")
    plot.lines[0].strokeWidth = 1.5
    plot.yValueAxis.valueMin = min(0.0, min(value for _, value in clear))
    plot.yValueAxis.valueMax = 1
    plot.yValueAxis.valueStep = 0.2
    plot.yValueAxis.labels.fontName = FONT
    plot.yValueAxis.labels.fontSize = 7
    plot.xValueAxis.labels.fontName = FONT
    plot.xValueAxis.labels.fontSize = 7
    plot.xValueAxis.labelTextFormat = lambda offset: (
        first + timedelta(days=int(offset))
    ).strftime("%d.%m")
    drawing.add(plot)
    drawing.add(
        String(18 * mm, 54 * mm, "NDVI по чистым снимкам", fontName=FONT, fontSize=8)
    )
    return drawing


def render_report(data: ReportData) -> bytes:
    _register_fonts()
    story: list[Flowable] = [
        Paragraph(f"Состояние участка «{data.name}»", TITLE),
        Paragraph(
            f"Отчёт VitiGuard MD на {_date(data.today)}, сформирован "
            f"{data.created_at.strftime('%d.%m.%Y %H:%M')}",
            SMALL,
        ),
        Spacer(1, 4 * mm),
    ]
    facts = [
        f"Площадь: {_number(data.area_ha)} га",
        f"Сорт: {data.variety or 'не указан'}",
        f"Год посадки: {data.planting_year or 'не указан'}",
    ]
    story.append(Paragraph(" · ".join(facts), TEXT))

    story.append(Paragraph("Риск болезней на неделю", HEADING))
    story.append(_risk_table(data.risk))

    story.append(Paragraph("Что делать", HEADING))
    protected = [
        f"{DISEASES[disease]}: "
        + (f"участок защищён до {_date(until)}" if until else "участок не защищён")
        for disease, until in data.plan.protected_until.items()
    ]
    story.append(Paragraph("; ".join(protected) + ".", TEXT))
    if data.plan.recommendations:
        for item in data.plan.recommendations:
            fungicide = (
                f", препарат: {FUNGICIDES[item.fungicide]}" if item.fungicide else ""
            )
            story.append(
                Paragraph(
                    f"• {ACTIONS[item.action]} — {DISEASES[item.disease].lower()}, "
                    f"до {_date(item.due_by)} (риск {_date(item.risk_day)}){fungicide}",
                    TEXT,
                )
            )
    else:
        story.append(Paragraph("Обработки на этой неделе не требуются.", TEXT))

    story.append(Paragraph("Спутниковые индексы сезона", HEADING))
    chart = _ndvi_chart(data.indices)
    if chart is not None:
        story.append(chart)
    if data.indices:
        rows = [["Снимок", "NDVI", "NDRE", "Облачность над участком"]]
        for point in data.indices[-RECENT:]:
            rows.append(
                [
                    _date(point.day),
                    _number(point.ndvi) if point.ndvi is not None else "—",
                    _number(point.ndre) if point.ndre is not None else "—",
                    f"{point.cloud:.0f} %" if point.cloud is not None else "—",
                ]
            )
        story.append(_table(rows, [35 * mm, 25 * mm, 25 * mm, 50 * mm]))
        story.append(Spacer(1, 2 * mm))
    else:
        story.append(Paragraph("Снимков за сезон пока нет.", TEXT))
    if data.anomalies:
        for anomaly in data.anomalies:
            kind = "подтверждённая" if anomaly.level == "confirmed" else "возможная"
            story.append(
                Paragraph(
                    f"• {_date(anomaly.acquired_at.date())}: {kind} просадка NDVI на "
                    f"{_number(anomaly.below_norm)} ниже нормы участка",
                    TEXT,
                )
            )
    else:
        story.append(Paragraph("Аномалий NDVI не найдено.", TEXT))

    story.append(Paragraph("Обработки", HEADING))
    if data.treatments:
        rows = [["Дата", "Тип препарата", "Препарат"]]
        rows += [
            [_date(t.applied_on), FUNGICIDES[t.fungicide], t.product or "—"]
            for t in data.treatments
        ]
        story.append(_table(rows, [35 * mm, 40 * mm, 90 * mm]))
    else:
        story.append(Paragraph("Обработок в журнале нет.", TEXT))

    story.append(Paragraph("Проверки листьев", HEADING))
    if data.checks:
        rows = [["Дата", "Диагноз модели", "Уверенность", "Заключение агронома"]]
        rows += [
            [
                _date(check.day),
                check.verdict,
                (
                    f"{check.confidence * 100:.0f} %"
                    if check.confidence is not None
                    else "—"
                ),
                check.expert or "—",
            ]
            for check in data.checks
        ]
        story.append(_table(rows, [30 * mm, 45 * mm, 28 * mm, 50 * mm]))
    else:
        story.append(Paragraph("Проверок по этому участку нет.", TEXT))

    story += [
        Spacer(1, 6 * mm),
        Paragraph(
            "Риск и рекомендации рассчитаны агромоделями по погоде Open-Meteo, индексы — "
            "по снимкам Sentinel-2. Это подсказка, а не заключение агронома; сроки "
            "защиты конкретного препарата — по его этикетке.",
            SMALL,
        ),
    ]

    buffer = io.BytesIO()
    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
        title=f"VitiGuard MD — {data.name}",
        author="VitiGuard MD",
    )
    document.build(story)
    return buffer.getvalue()
