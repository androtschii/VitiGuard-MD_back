"""Ответы бота о своём хозяйстве и справка о болезнях.

Риск на сегодня, план обработок и последние проверки берутся из тех же
расчётов, что показывает сайт: бот только собирает их в короткое сообщение.
Вопрос можно задать командой или своими словами."""

import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.diagnosis import DiseaseResult, LeafImage
from app.models.enums import Disease, Fungicide, ProcessingStatus, RiskLevel
from app.models.vineyard import Vineyard
from app.services.agromodels import LOCAL_TZ
from app.services.risk import TodayLevels, local_today, today_levels
from app.services.treatments import Action, Plan, vineyard_plan

HISTORY_LIMIT = 5
# План по каждому участку — отдельный расчёт: в сообщение идут первые участки
PLAN_VINEYARDS = 10

LEVELS = {
    RiskLevel.HIGH: "🔴 высокий",
    RiskLevel.MEDIUM: "🟡 средний",
    RiskLevel.LOW: "🟢 низкий",
}
ACTIONS = {
    Action.PROTECT: "защитная обработка",
    Action.CURE: "лечебная обработка",
    Action.INSPECT: "осмотр кустов",
}
DISEASES = {
    Disease.DOWNY_MILDEW: "милдью",
    Disease.POWDERY_MILDEW: "оидиума",
}
FUNGICIDES = {
    Fungicide.CONTACT: "контактным препаратом (медь)",
    Fungicide.SYSTEMIC: "системным препаратом",
    Fungicide.SULFUR: "препаратом серы",
}
REASONS = {
    "downy_infection_past": "было заражение милдью {day}",
    "downy_infection_forecast": "по прогнозу заражение милдью {day}",
    "downy_symptoms": "ожидаются пятна милдью после заражения",
    "powdery_high": "высокий индекс оидиума {day}",
    "powdery_high_hot": "высокий индекс оидиума {day}, а в жару сера обжигает листья",
}
CLASS_NAMES = {
    "healthy": "здоровый лист",
    "black_rot": "чёрная гниль",
    "esca": "эска",
    "leaf_blight": "пятнистость листьев",
    "downy_mildew": "милдью",
    "powdery_mildew": "оидиум",
}

NO_VINEYARDS = "Участков пока нет: нарисуйте их на карте в VitiGuard MD."
NOTE = (
    "Препараты и дозы подбирайте по государственному реестру разрешённых "
    "средств защиты растений и с агрономом."
)


@dataclass(frozen=True)
class DiseaseInfo:
    name: str
    # Основы слов, по которым болезнь узнаётся в вопросе
    words: tuple[str, ...]
    symptoms: str
    conditions: str
    control: str


DISEASE_INFO = {
    "downy_mildew": DiseaseInfo(
        "Милдью",
        ("милдью", "ложная мучнист", "переноспор"),
        "Сверху листа — жёлтые «масляные» пятна, снизу под ними — белый "
        "пушистый налёт; поражает соцветия и молодые ягоды.",
        "Дождь и влажный лист при температуре от 10 °C — правило «трёх "
        "десяток»; после заражения пятна видны через 5-18 дней.",
        "Контактные (медные) препараты — до дождя, системные — в первые дни "
        "после заражения; удаляйте пасынки. Риск по участкам — /risk.",
    ),
    "powdery_mildew": DiseaseInfo(
        "Оидиум",
        ("оидиум", "мучнист"),
        "Серо-белый мучнистый налёт с обеих сторон листа и на ягодах, запах "
        "плесени; ягоды трескаются.",
        "Тепло 20-30 °C и влажный воздух без дождя, загущённая крона.",
        "Препараты серы и системные фунгициды с интервалом по индексу "
        "Gubler-Thomas; зелёные операции и проветривание кроны.",
    ),
    "black_rot": DiseaseInfo(
        "Чёрная гниль",
        ("чёрная гниль", "черная гниль", "чёрной гнил", "черной гнил"),
        "Округлые бурые пятна с тёмной каймой и чёрными точками внутри; "
        "ягоды буреют и сморщиваются в чёрные «мумии».",
        "Тёплая влажная погода и дожди от распускания почек до смыкания ягод.",
        "Уберите «мумии» и поражённые побеги, проветривайте крону; "
        "профилактические обработки от цветения до смыкания ягод.",
    ),
    "esca": DiseaseInfo(
        "Эска",
        ("эска", "эски", "эску"),
        "«Тигровые» полосы между жилками — сначала жёлтые или красные, потом "
        "бурые; в жару куст может внезапно усохнуть.",
        "Болезнь древесины: грибы проникают через раны от обрезки, чаще на "
        "старых кустах.",
        "Химического лечения нет. Обрезайте в сухую погоду и закрывайте "
        "крупные срезы, вырезайте и сжигайте поражённые рукава.",
    ),
    "leaf_blight": DiseaseInfo(
        "Пятнистость листьев",
        ("пятнистост",),
        "Неровные бурые пятна на листе, позже сливаются; листья рано опадают.",
        "Тёплая влажная погода во второй половине лета.",
        "Убирайте опавшие листья, проветривайте крону; обычно хватает "
        "обработок против милдью и чёрной гнили.",
    ),
}
DISEASES_TEXT = (
    "Справка о болезнях винограда: напишите название — милдью, оидиум, "
    "чёрная гниль, эска или пятнистость листьев."
)

# Вопросы своими словами о своём хозяйстве
RISK_WORDS = ("риск", "опасн", "угроз")
PLAN_WORDS = ("план", "когда обрабат", "защищ")
HISTORY_WORDS = ("истори", "последн", "мои провер")


def short_date(day: date) -> str:
    return day.strftime("%d.%m")


def farm_topic(text: str) -> str | None:
    """Команда для вопроса своими словами: /risk, /plan или /history."""
    question = text.lower()
    if any(word in question for word in PLAN_WORDS):
        return "/plan"
    if any(word in question for word in RISK_WORDS):
        return "/risk"
    if any(word in question for word in HISTORY_WORDS):
        return "/history"
    return None


def disease_reply(text: str) -> str | None:
    question = text.lower()
    for info in DISEASE_INFO.values():
        if any(word in question for word in info.words):
            return (
                f"{info.name}\n\n"
                f"Признаки: {info.symptoms}\n\n"
                f"Когда развивается: {info.conditions}\n\n"
                f"Что делать: {info.control}\n\n{NOTE}"
            )
    return None


def risk_text(
    vineyards: Sequence[Vineyard], levels: Sequence[TodayLevels], today: date
) -> str:
    if not vineyards:
        return NO_VINEYARDS
    by_vineyard = {item.vineyard_id: item for item in levels}
    lines = [f"Риск болезней на сегодня, {short_date(today)}:"]
    for vineyard in vineyards:
        item = by_vineyard.get(vineyard.id)
        if item is None:
            lines.append(f"• {vineyard.name}: оценки на сегодня нет")
        else:
            lines.append(
                f"• {vineyard.name}: милдью {LEVELS[item.downy]}, "
                f"оидиум {LEVELS[item.powdery]}"
            )
    if len(by_vineyard) < len(vineyards):
        lines.append("Оценки нет вне сезона или если риск участка ещё не пересчитан.")
    return "\n".join(lines)


def plan_lines(name: str, plan: Plan) -> list[str]:
    lines = [f"{name}:"]
    for disease, until in plan.protected_until.items():
        if until is not None:
            lines.append(f"  защищён от {DISEASES[disease]} до {short_date(until)}")
    if not plan.recommendations:
        lines.append("  обработки на неделю не нужны")
    for item in plan.recommendations:
        what = ACTIONS[item.action]
        if item.action is not Action.INSPECT:
            what += f" от {DISEASES[item.disease]}"
        if item.fungicide is not None:
            what += f" {FUNGICIDES[item.fungicide]}"
        reason = REASONS.get(item.reason, "").format(day=short_date(item.risk_day))
        lines.append(
            f"  до {short_date(item.due_by)}: {what}"
            + (f" — {reason}" if reason else "")
        )
    return lines


def plan_text(plans: Sequence[tuple[str, Plan]], total: int) -> str:
    if not plans:
        return NO_VINEYARDS
    lines = ["План обработок на неделю:"]
    for name, plan in plans:
        lines.extend(plan_lines(name, plan))
    if total > len(plans):
        lines.append(f"Ещё участков: {total - len(plans)} — план есть на сайте.")
    lines.append(NOTE)
    return "\n".join(lines)


@dataclass(frozen=True)
class CheckRow:
    created_at: datetime
    predicted_class: str
    confidence: float
    vineyard: str | None
    expert_class: str | None


def history_text(rows: Sequence[CheckRow]) -> str:
    if not rows:
        return (
            "Сохранённых проверок пока нет: пришлите фото листа — "
            "бот определит болезнь и сохранит проверку."
        )
    lines = ["Последние проверки листьев:"]
    for row in rows:
        day = row.created_at.astimezone(LOCAL_TZ).strftime("%d.%m %H:%M")
        name = CLASS_NAMES.get(row.predicted_class, row.predicted_class)
        place = f", {row.vineyard}" if row.vineyard else ""
        line = f"• {day}{place}: {name}, {round(row.confidence * 100)} %"
        if row.expert_class is not None:
            expert = CLASS_NAMES.get(row.expert_class, row.expert_class)
            line += f" (агроном: {expert})"
        lines.append(line)
    return "\n".join(lines)


async def owner_vineyards(session: AsyncSession, owner_id: uuid.UUID) -> list[Vineyard]:
    rows = await session.scalars(
        select(Vineyard).where(Vineyard.owner_id == owner_id).order_by(Vineyard.name)
    )
    return list(rows)


async def farm_reply(session: AsyncSession, owner_id: uuid.UUID, command: str) -> str:
    today = local_today()
    if command == "/history":
        result = await session.execute(
            select(
                DiseaseResult.created_at,
                DiseaseResult.predicted_class,
                DiseaseResult.confidence,
                Vineyard.name,
                DiseaseResult.expert_class,
            )
            .join(LeafImage, LeafImage.id == DiseaseResult.image_id)
            .outerjoin(Vineyard, Vineyard.id == LeafImage.vineyard_id)
            .where(
                LeafImage.owner_id == owner_id,
                DiseaseResult.status == ProcessingStatus.COMPLETED,
            )
            .order_by(DiseaseResult.created_at.desc())
            .limit(HISTORY_LIMIT)
        )
        return history_text(
            [
                CheckRow(created, cls, confidence, vineyard, expert)
                for created, cls, confidence, vineyard, expert in result
                if cls is not None and confidence is not None
            ]
        )
    vineyards = await owner_vineyards(session, owner_id)
    if command == "/risk":
        return risk_text(vineyards, await today_levels(session, owner_id, today), today)
    plans = [
        (vineyard.name, await vineyard_plan(session, vineyard.id, today))
        for vineyard in vineyards[:PLAN_VINEYARDS]
    ]
    return plan_text(plans, len(vineyards))
