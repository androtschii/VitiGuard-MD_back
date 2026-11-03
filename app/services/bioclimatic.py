"""Биоклиматические индексы терруара, фенология BBCH и прогноз окна сбора урожая.

Моделирование термических условий виноградарства:
1. Активные температуры (GDD, Growing Degree Days, база 10 °C).
2. Фенологическая модель фаз развития по международной шкале BBCH (00-99)
   с калибровкой для ключевых сортов винограда Молдовы.
3. Биоклиматические индексы классификации терруара:
   - Индекс Винклера (Winkler Index, сумма активных температур за апрель–октябрь).
   - Гелиотермический индекс Гуглина (Huglin Index, с учётом широты Молдовы d=1.04).
   - Индекс прохладных ночей (Cool Night Index, средний минимум сентября).
   - Гидротермический коэффициент Селянинова (ГТК, баланс осадков и тепла).
4. Прогностическая модель окна сбора урожая на основе текущего сахара (°Brix),
   накопленной суммы температур и 14-дневного прогноза погоды с анализом агрорисков.
"""

import enum
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any

from app.services.agromodels import GDD_BASE, DailyWeather

MOLDOVA_LATITUDE = 47.0
HUGLIN_LATITUDE_FACTOR_MOLDOVA = 1.04

WINKLER_START = (4, 1)
WINKLER_END = (10, 31)

HUGLIN_START = (4, 1)
HUGLIN_END = (9, 30)

SEPTEMBER_MONTH = 9


class BBCHStage(enum.IntEnum):
    """Международная шкала фенологических фаз винограда BBCH (00-99)."""

    DORMANCY = 0  # Покой (зимняя почка)
    BUDBURST = 9  # Распускание почек (зелёный кончик)
    SHOOT_GROWTH = 15  # Развитие побегов (5-6 листьев развёрнуто)
    INFLORESCENCE = 55  # Обособление соцветий
    FLOWERING = 65  # Полное цветение (50% чехликов опало)
    FRUIT_SET = 71  # Завязывание ягод
    BERRIES_PEA_SIZED = 75  # Горошина / смыкание ягод
    VERAISON = 81  # Начало созревания (веaison, окрашивание ягод)
    BERRY_SOFTENING = 85  # Размягчение ягод
    HARVEST_MATURITY = 89  # Техническая зрелость (оптимальный сбор)
    LEAF_FALL = 93  # Начало листопада


BBCH_NAMES: dict[BBCHStage, str] = {
    BBCHStage.DORMANCY: "Покой (зимняя почка)",
    BBCHStage.BUDBURST: "Распускание почек (зелёный кончик)",
    BBCHStage.SHOOT_GROWTH: "Развитие побегов и листьев (5-6 листьев)",
    BBCHStage.INFLORESCENCE: "Обособление соцветий",
    BBCHStage.FLOWERING: "Полное цветение",
    BBCHStage.FRUIT_SET: "Завязывание ягод",
    BBCHStage.BERRIES_PEA_SIZED: "Смыкание ягод (размер «горошина»)",
    BBCHStage.VERAISON: "Начало созревания (веaison)",
    BBCHStage.BERRY_SOFTENING: "Размягчение ягод",
    BBCHStage.HARVEST_MATURITY: "Техническая зрелость (сбор урожая)",
    BBCHStage.LEAF_FALL: "Начало листопада",
}


@dataclass(frozen=True)
class VarietyThermalProfile:
    """Термический профиль сорта: требуемые суммы активных температур (GDD)."""

    name: str
    code: str
    berry_color: str
    gdd_thresholds: dict[BBCHStage, float]
    target_brix_default: float
    description: str


# Калибровка по исследованиям Института виноградарства Молдовы (Кишинёв)
# и европейским ампелографическим атласам
VARIETY_CATALOG: dict[str, VarietyThermalProfile] = {
    "feteasca_regala": VarietyThermalProfile(
        name="Fetească Regală",
        code="feteasca_regala",
        berry_color="white",
        gdd_thresholds={
            BBCHStage.BUDBURST: 110.0,
            BBCHStage.SHOOT_GROWTH: 220.0,
            BBCHStage.INFLORESCENCE: 350.0,
            BBCHStage.FLOWERING: 420.0,
            BBCHStage.FRUIT_SET: 520.0,
            BBCHStage.BERRIES_PEA_SIZED: 750.0,
            BBCHStage.VERAISON: 980.0,
            BBCHStage.BERRY_SOFTENING: 1150.0,
            BBCHStage.HARVEST_MATURITY: 1320.0,
        },
        target_brix_default=20.5,
        description="Автохтонный белый сорт средне-раннего созревания с высокой кислотностью.",
    ),
    "feteasca_alba": VarietyThermalProfile(
        name="Fetească Albă",
        code="feteasca_alba",
        berry_color="white",
        gdd_thresholds={
            BBCHStage.BUDBURST: 105.0,
            BBCHStage.SHOOT_GROWTH: 210.0,
            BBCHStage.INFLORESCENCE: 340.0,
            BBCHStage.FLOWERING: 410.0,
            BBCHStage.FRUIT_SET: 510.0,
            BBCHStage.BERRIES_PEA_SIZED: 730.0,
            BBCHStage.VERAISON: 950.0,
            BBCHStage.BERRY_SOFTENING: 1120.0,
            BBCHStage.HARVEST_MATURITY: 1280.0,
        },
        target_brix_default=20.0,
        description="Тонкий автохтонный ранний сорт для свежих тихих и игристых вин.",
    ),
    "viorica": VarietyThermalProfile(
        name="Viorica",
        code="viorica",
        berry_color="white",
        gdd_thresholds={
            BBCHStage.BUDBURST: 115.0,
            BBCHStage.SHOOT_GROWTH: 230.0,
            BBCHStage.INFLORESCENCE: 360.0,
            BBCHStage.FLOWERING: 430.0,
            BBCHStage.FRUIT_SET: 530.0,
            BBCHStage.BERRIES_PEA_SIZED: 770.0,
            BBCHStage.VERAISON: 1020.0,
            BBCHStage.BERRY_SOFTENING: 1180.0,
            BBCHStage.HARVEST_MATURITY: 1350.0,
        },
        target_brix_default=21.0,
        description="Молдавский селекционный ароматный сорт группы Мускатов с тонами базилика и полевых трав.",
    ),
    "chardonnay": VarietyThermalProfile(
        name="Chardonnay",
        code="chardonnay",
        berry_color="white",
        gdd_thresholds={
            BBCHStage.BUDBURST: 110.0,
            BBCHStage.SHOOT_GROWTH: 220.0,
            BBCHStage.INFLORESCENCE: 350.0,
            BBCHStage.FLOWERING: 420.0,
            BBCHStage.FRUIT_SET: 520.0,
            BBCHStage.BERRIES_PEA_SIZED: 750.0,
            BBCHStage.VERAISON: 1000.0,
            BBCHStage.BERRY_SOFTENING: 1170.0,
            BBCHStage.HARVEST_MATURITY: 1350.0,
        },
        target_brix_default=22.0,
        description="Классический белый сорт с умеренными требованиями к сумме активных температур.",
    ),
    "sauvignon_blanc": VarietyThermalProfile(
        name="Sauvignon Blanc",
        code="sauvignon_blanc",
        berry_color="white",
        gdd_thresholds={
            BBCHStage.BUDBURST: 120.0,
            BBCHStage.SHOOT_GROWTH: 240.0,
            BBCHStage.INFLORESCENCE: 370.0,
            BBCHStage.FLOWERING: 440.0,
            BBCHStage.FRUIT_SET: 540.0,
            BBCHStage.BERRIES_PEA_SIZED: 790.0,
            BBCHStage.VERAISON: 1050.0,
            BBCHStage.BERRY_SOFTENING: 1220.0,
            BBCHStage.HARVEST_MATURITY: 1420.0,
        },
        target_brix_default=21.5,
        description="Сорт среднего срока созревания с выраженным пиразиновым и тиоловым профилем.",
    ),
    "feteasca_neagra": VarietyThermalProfile(
        name="Fetească Neagră",
        code="feteasca_neagra",
        berry_color="red",
        gdd_thresholds={
            BBCHStage.BUDBURST: 125.0,
            BBCHStage.SHOOT_GROWTH: 250.0,
            BBCHStage.INFLORESCENCE: 380.0,
            BBCHStage.FLOWERING: 460.0,
            BBCHStage.FRUIT_SET: 560.0,
            BBCHStage.BERRIES_PEA_SIZED: 830.0,
            BBCHStage.VERAISON: 1150.0,
            BBCHStage.BERRY_SOFTENING: 1350.0,
            BBCHStage.HARVEST_MATURITY: 1550.0,
        },
        target_brix_default=23.5,
        description="Благородный автохтонный красный сорт позднего срока созревания для глубоких выдержанных вин.",
    ),
    "rara_neagra": VarietyThermalProfile(
        name="Rară Neagră",
        code="rara_neagra",
        berry_color="red",
        gdd_thresholds={
            BBCHStage.BUDBURST: 130.0,
            BBCHStage.SHOOT_GROWTH: 260.0,
            BBCHStage.INFLORESCENCE: 400.0,
            BBCHStage.FLOWERING: 480.0,
            BBCHStage.FRUIT_SET: 580.0,
            BBCHStage.BERRIES_PEA_SIZED: 860.0,
            BBCHStage.VERAISON: 1200.0,
            BBCHStage.BERRY_SOFTENING: 1400.0,
            BBCHStage.HARVEST_MATURITY: 1620.0,
        },
        target_brix_default=22.5,
        description="Старинный молдавский красный сорт с яркой гранатовой кислотностью и мягким танином.",
    ),
    "merlot": VarietyThermalProfile(
        name="Merlot",
        code="merlot",
        berry_color="red",
        gdd_thresholds={
            BBCHStage.BUDBURST: 120.0,
            BBCHStage.SHOOT_GROWTH: 240.0,
            BBCHStage.INFLORESCENCE: 370.0,
            BBCHStage.FLOWERING: 450.0,
            BBCHStage.FRUIT_SET: 550.0,
            BBCHStage.BERRIES_PEA_SIZED: 810.0,
            BBCHStage.VERAISON: 1120.0,
            BBCHStage.BERRY_SOFTENING: 1310.0,
            BBCHStage.HARVEST_MATURITY: 1520.0,
        },
        target_brix_default=23.0,
        description="Красный сорт среднего срока созревания с округлым бархатным телом.",
    ),
    "cabernet_sauvignon": VarietyThermalProfile(
        name="Cabernet Sauvignon",
        code="cabernet_sauvignon",
        berry_color="red",
        gdd_thresholds={
            BBCHStage.BUDBURST: 135.0,
            BBCHStage.SHOOT_GROWTH: 270.0,
            BBCHStage.INFLORESCENCE: 410.0,
            BBCHStage.FLOWERING: 490.0,
            BBCHStage.FRUIT_SET: 600.0,
            BBCHStage.BERRIES_PEA_SIZED: 890.0,
            BBCHStage.VERAISON: 1250.0,
            BBCHStage.BERRY_SOFTENING: 1450.0,
            BBCHStage.HARVEST_MATURITY: 1680.0,
        },
        target_brix_default=24.0,
        description="Поздний теплолюбивый красный сорт с мощной структурой и танинами.",
    ),
    "pinot_noir": VarietyThermalProfile(
        name="Pinot Noir",
        code="pinot_noir",
        berry_color="red",
        gdd_thresholds={
            BBCHStage.BUDBURST: 110.0,
            BBCHStage.SHOOT_GROWTH: 220.0,
            BBCHStage.INFLORESCENCE: 350.0,
            BBCHStage.FLOWERING: 420.0,
            BBCHStage.FRUIT_SET: 520.0,
            BBCHStage.BERRIES_PEA_SIZED: 760.0,
            BBCHStage.VERAISON: 1020.0,
            BBCHStage.BERRY_SOFTENING: 1190.0,
            BBCHStage.HARVEST_MATURITY: 1360.0,
        },
        target_brix_default=22.0,
        description="Ранний тонкий красный сорт, требовательный к прохладным ночам.",
    ),
}

DEFAULT_PROFILE = VARIETY_CATALOG["feteasca_regala"]


def get_variety_profile(variety_identifier: str | None) -> VarietyThermalProfile:
    """Возвращает профиль сорта по его системному коду или названию."""
    if not variety_identifier:
        return DEFAULT_PROFILE
    normalized = (
        variety_identifier.strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
        .replace("ă", "a")
        .replace("î", "i")
        .replace("ș", "s")
        .replace("ț", "t")
    )
    for code, profile in VARIETY_CATALOG.items():
        if (
            normalized == code
            or normalized in profile.name.lower()
            or code in normalized
        ):
            return profile
    return DEFAULT_PROFILE


@dataclass(frozen=True)
class PhenologyEstimate:
    """Оценка текущей фенологической фазы лозы."""

    current_bbch: int
    current_stage_name: str
    accumulated_gdd: float
    next_bbch: int | None
    next_stage_name: str | None
    gdd_to_next: float
    progress_percentage: float
    is_season_complete: bool


def estimate_phenology_stage(
    accumulated_gdd: float,
    variety: str | None = None,
) -> PhenologyEstimate:
    """Определяет текущую фазу BBCH по сумме активных температур."""
    profile = get_variety_profile(variety)
    sorted_stages = sorted(profile.gdd_thresholds.items(), key=lambda x: x[1])

    if accumulated_gdd < sorted_stages[0][1]:
        # До распускания почек
        current_bbch = int(BBCHStage.DORMANCY)
        next_bbch = int(sorted_stages[0][0])
        next_name = BBCH_NAMES[sorted_stages[0][0]]
        gdd_to_next = round(sorted_stages[0][1] - accumulated_gdd, 1)
        progress = round(
            max(0.0, min(100.0, (accumulated_gdd / sorted_stages[0][1]) * 100.0)), 1
        )
        return PhenologyEstimate(
            current_bbch=current_bbch,
            current_stage_name=BBCH_NAMES[BBCHStage.DORMANCY],
            accumulated_gdd=round(accumulated_gdd, 1),
            next_bbch=next_bbch,
            next_stage_name=next_name,
            gdd_to_next=gdd_to_next,
            progress_percentage=progress,
            is_season_complete=False,
        )

    current_stage = sorted_stages[0][0]
    next_stage: BBCHStage | None = None
    stage_start_gdd = 0.0
    stage_end_gdd = sorted_stages[0][1]

    for i, (stage, threshold) in enumerate(sorted_stages):
        if accumulated_gdd >= threshold:
            current_stage = stage
            stage_start_gdd = threshold
            if i + 1 < len(sorted_stages):
                next_stage = sorted_stages[i + 1][0]
                stage_end_gdd = sorted_stages[i + 1][1]
            else:
                next_stage = None
                stage_end_gdd = threshold

    if next_stage is None:
        return PhenologyEstimate(
            current_bbch=int(current_stage),
            current_stage_name=BBCH_NAMES[current_stage],
            accumulated_gdd=round(accumulated_gdd, 1),
            next_bbch=None,
            next_stage_name=None,
            gdd_to_next=0.0,
            progress_percentage=100.0,
            is_season_complete=True,
        )

    span = stage_end_gdd - stage_start_gdd
    progress = (
        0.0 if span <= 0 else ((accumulated_gdd - stage_start_gdd) / span) * 100.0
    )
    gdd_needed = max(0.0, stage_end_gdd - accumulated_gdd)

    return PhenologyEstimate(
        current_bbch=int(current_stage),
        current_stage_name=BBCH_NAMES[current_stage],
        accumulated_gdd=round(accumulated_gdd, 1),
        next_bbch=int(next_stage),
        next_stage_name=BBCH_NAMES[next_stage],
        gdd_to_next=round(gdd_needed, 1),
        progress_percentage=round(max(0.0, min(100.0, progress)), 1),
        is_season_complete=False,
    )


@dataclass(frozen=True)
class WinklerResult:
    """Индекс тепла Винклера (Amerine & Winkler, GDD с 1 апреля по 31 октября)."""

    gdd_sum: float
    region_class: str
    region_title: str
    suitable_varieties: list[str]


def compute_winkler_index(days: Sequence[DailyWeather]) -> WinklerResult:
    """Вычисляет индекс Винклера за вегетационный период (01.04 - 31.10)."""
    total = 0.0
    for day in days:
        d = day.day
        if date(d.year, *WINKLER_START) <= d <= date(d.year, *WINKLER_END):
            total += max(0.0, day.mean_temp - GDD_BASE)

    total = round(total, 1)

    if total <= 1390:
        return WinklerResult(
            gdd_sum=total,
            region_class="Region I",
            region_title="Прохладный климат (Very Cool / Cool)",
            suitable_varieties=[
                "Chardonnay (база для игристых)",
                "Pinot Noir",
                "Riesling",
                "Fetească Albă",
            ],
        )
    if total <= 1670:
        return WinklerResult(
            gdd_sum=total,
            region_class="Region II",
            region_title="Умеренный климат (Moderate)",
            suitable_varieties=[
                "Sauvignon Blanc",
                "Chardonnay",
                "Fetească Regală",
                "Merlot",
                "Viorica",
            ],
        )
    if total <= 1940:
        return WinklerResult(
            gdd_sum=total,
            region_class="Region III",
            region_title="Тёплый климат (Warm)",
            suitable_varieties=[
                "Cabernet Sauvignon",
                "Merlot",
                "Fetească Neagră",
                "Rară Neagră",
                "Syrah",
            ],
        )
    if total <= 2220:
        return WinklerResult(
            gdd_sum=total,
            region_class="Region IV",
            region_title="Жаркий климат (Hot)",
            suitable_varieties=[
                "Saperavi",
                "Cabernet Sauvignon",
                "Malbec",
                "десертные и креплёные вина",
            ],
        )
    return WinklerResult(
        gdd_sum=total,
        region_class="Region V",
        region_title="Очень жаркий климат (Very Hot)",
        suitable_varieties=["Столовые сорта", "креплёные вина"],
    )


@dataclass(frozen=True)
class HuglinResult:
    """Гелиотермический индекс Гуглина (Huglin, 1978).

    HI = sum_{01.04}^{30.09} [ (T_mean - 10) + (T_max - 10) ] / 2 * d
    d = 1.04 для широт Молдовы 46-48°N.
    """

    hi_value: float
    thermal_class: str
    thermal_description: str


def compute_huglin_index(
    days: Sequence[DailyWeather],
    latitude: float = MOLDOVA_LATITUDE,
) -> HuglinResult:
    """Вычисляет гелиотермический индекс Гуглина."""
    if latitude < 40.0:
        d = 1.00
    elif latitude < 44.0:
        d = 1.02
    elif latitude <= 48.0:
        d = HUGLIN_LATITUDE_FACTOR_MOLDOVA
    else:
        d = 1.06

    total = 0.0
    for day in days:
        dt = day.day
        if date(dt.year, *HUGLIN_START) <= dt <= date(dt.year, *HUGLIN_END):
            t_mean_diff = max(0.0, day.mean_temp - GDD_BASE)
            t_max_diff = max(0.0, day.max_temp - GDD_BASE)
            daily_heat = ((t_mean_diff + t_max_diff) / 2.0) * d
            total += daily_heat

    hi = round(total, 1)

    if hi <= 1500:
        t_class = "HI-1 (Очень прохладный)"
        t_desc = "Подходит только для ультраранних сортов и базовых игристых."
    elif hi <= 1800:
        t_class = "HI-2 (Прохладный)"
        t_desc = "Идеален для ароматных белых сортов: Fetească Albă, Sauvignon Blanc, Pinot Grigio."
    elif hi <= 2100:
        t_class = "HI-3 (Умеренно-тёплый)"
        t_desc = "Оптимум для большинства молдавских сортов: Fetească Regală, Viorica, Merlot."
    elif hi <= 2400:
        t_class = "HI-4 (Тёплый)"
        t_desc = "Отлично вызревают позднеспелые красные сорта: Fetească Neagră, Cabernet Sauvignon, Rară Neagră."
    elif hi <= 3000:
        t_class = "HI-5 (Очень тёплый)"
        t_desc = "Высокая сахаронакопительная способность, риск снижения кислотности."
    else:
        t_class = "HI-6 (Слишком жаркий)"
        t_desc = "Экстремально высокая теплообеспеченность, риск увядания ягод."

    return HuglinResult(hi_value=hi, thermal_class=t_class, thermal_description=t_desc)


@dataclass(frozen=True)
class CoolNightResult:
    """Индекс прохладных ночей (Tonietto & Carbonneau, 2004, CI).

    Средняя минимальная температура сентября (T_min_september).
    Критически важен для сохранения свежих тиолов, терпенов и синтеза антоцианов в кожице ягод.
    """

    ci_temperature: float
    night_class: str
    aroma_color_potential: str


def compute_cool_night_index(days: Sequence[DailyWeather]) -> CoolNightResult:
    """Вычисляет индекс прохладных ночей (CI) по сентябрьским дням."""
    september_mins = [day.min_temp for day in days if day.day.month == SEPTEMBER_MONTH]

    if not september_mins:
        return CoolNightResult(
            ci_temperature=0.0,
            night_class="Нет данных за сентябрь",
            aroma_color_potential="Требуются данные за сентябрь.",
        )

    ci = round(sum(september_mins) / len(september_mins), 1)

    if ci <= 12.0:
        n_class = "CI+2 (Очень прохладные ночи)"
        n_desc = (
            "Идеальная фиксация антоцианов и свежей яблочной/винной кислоты. "
            "Яркие свежие тона белых и рубиновый цвет красных вин."
        )
    elif ci <= 14.0:
        n_class = "CI+1 (Прохладные ночи)"
        n_desc = "Прекрасный баланс созревания сахаров, фенолов и ароматических предшественников."
    elif ci <= 18.0:
        n_class = "CI-1 (Умеренно тёплые ночи)"
        n_desc = "Ускоренная деградация яблочной кислоты, мягкие зрелые танины, умеренная кислотность."
    else:
        n_class = "CI-2 (Тёплые ночи)"
        n_desc = "Риск плоских вин с недостаточной кислотностью и бледным цветом красных сортов."

    return CoolNightResult(
        ci_temperature=ci, night_class=n_class, aroma_color_potential=n_desc
    )


@dataclass(frozen=True)
class SelyaninovResult:
    """Гидротермический коэффициент Селянинова (ГТК).

    ГТК = sum(P) / [ 0.1 * sum(T_mean) ] для дней с T_mean >= 10 °C.
    Показывает засушливость и потребность в орошении виноградника.
    """

    htc_value: float
    moisture_class: str
    irrigation_need: str


def compute_selyaninov_htc(days: Sequence[DailyWeather]) -> SelyaninovResult:
    """Вычисляет ГТК Селянинова для активного вегетационного периода."""
    active_days = [day for day in days if day.mean_temp >= GDD_BASE]
    if not active_days:
        return SelyaninovResult(
            htc_value=0.0,
            moisture_class="Недостаточно активных дней",
            irrigation_need="Данные отсутствуют",
        )

    sum_precipitation = sum(day.precipitation_mm for day in active_days)
    sum_temperature = sum(day.mean_temp for day in active_days)

    if sum_temperature <= 0:
        return SelyaninovResult(
            htc_value=0.0,
            moisture_class="Нулевая сумма температур",
            irrigation_need="Не определено",
        )

    htc = round(sum_precipitation / (0.1 * sum_temperature), 2)

    if htc < 0.4:
        m_class = "Очень засушливая зона (острая аридность)"
        irr = "Критически необходимо капельное орошение для предотвращения сброса листьев и остановки созревания."
    elif htc < 0.7:
        m_class = "Засушливая зона"
        irr = "Рекомендуется регулярное поддерживающее орошение в период роста ягод (BBCH 71-79)."
    elif htc < 1.0:
        m_class = "Слабозасушливая зона (умеренный дефицит)"
        irr = "Оптимальный лёгкий водный стресс для качества красных сортов; орошение по датчикам влажности почвы."
    elif htc <= 1.4:
        m_class = "Зона достаточного увлажнения"
        irr = "Орошение не требуется; повышенное внимание к профилактике грибковых заболеваний (милдью, ботритис)."
    else:
        m_class = "Зона избыточного увлажнения"
        irr = "Высокий риск гнилей гроздей и растрескивания ягод; требуется залужение междурядий и дренаж."

    return SelyaninovResult(htc_value=htc, moisture_class=m_class, irrigation_need=irr)


@dataclass(frozen=True)
class HarvestWindowPrediction:
    """Прогноз оптимального окна сбора урожая."""

    variety_name: str
    target_sugar_brix: float
    predicted_brix_trajectory: list[dict[str, Any]]
    optimal_window_start: date | None
    optimal_window_end: date | None
    harvest_urgency: str
    risks_detected: list[str]
    agronomic_recommendation: str


def predict_harvest_window(
    accumulated_gdd: float,
    forecast_days: Sequence[DailyWeather],
    variety: str | None = None,
    current_sugar_brix: float | None = None,
    target_sugar_brix: float | None = None,
) -> HarvestWindowPrediction:
    """Моделирует динамику накопления сахаров и рекомендует окно сбора урожая.

    Учитывает:
    - Физиологический оптимум сахаронакопления (18-26 °C).
    - Торможение фотосинтеза при температуре выше 33 °C.
    - Разбавление сахаров и риск серой гнили (Botrytis cinerea) при осадках > 15 мм.
    - Опасность осенних заморозков (T_min < 0 °C).
    """
    profile = get_variety_profile(variety)
    target_brix = (
        target_sugar_brix
        if target_sugar_brix is not None
        else profile.target_brix_default
    )

    if current_sugar_brix is not None:
        initial_brix = current_sugar_brix
    else:
        # Приближение текущего Brix по накопленной сумме температур
        maturity_gdd = profile.gdd_thresholds[BBCHStage.HARVEST_MATURITY]
        veraison_gdd = profile.gdd_thresholds[BBCHStage.VERAISON]
        if accumulated_gdd <= veraison_gdd:
            initial_brix = 10.0
        elif accumulated_gdd >= maturity_gdd:
            initial_brix = target_brix
        else:
            fraction = (accumulated_gdd - veraison_gdd) / max(
                1.0, maturity_gdd - veraison_gdd
            )
            initial_brix = round(10.0 + fraction * (target_brix - 10.0), 1)

    trajectory: list[dict[str, Any]] = []
    current_brix = initial_brix
    risks: list[str] = []

    optimal_start: date | None = None
    optimal_end: date | None = None

    for day in forecast_days:
        # Базовая скорость накопления: 0.28 °Brix в сутки при идеальных условиях
        base_rate = 0.28

        # Температурный коэффициент активности ферментов
        t = day.mean_temp
        if t < 12.0:
            temp_factor = 0.15
        elif t <= 26.0:
            # Оптимум 18-26 °C
            temp_factor = 1.0 if t >= 18.0 else 0.4 + 0.6 * ((t - 12.0) / 6.0)
        elif t <= 33.0:
            temp_factor = 0.8
        else:
            # Стресс от жары: закрытие устьиц листьев
            temp_factor = 0.3
            if (
                f"Жара {day.max_temp:.1f} °C на {day.day}: торможение фотосинтеза и риск деградации кислот"
                not in risks
            ):
                risks.append(
                    f"Жара {day.max_temp:.1f} °C на {day.day}: торможение фотосинтеза и риск деградации кислот"
                )

        # Влияние осадков
        rain_penalty = 0.0
        if day.precipitation_mm >= 15.0:
            rain_penalty = 0.35  # Водопоглощение корнями и тургор ягод
            warning = f"Обильные осадки {day.precipitation_mm:.1f} мм на {day.day}: риск растрескивания ягод и серой гнили"
            if warning not in risks:
                risks.append(warning)
        elif day.precipitation_mm >= 5.0:
            rain_penalty = 0.1

        # Заморозки
        if day.min_temp <= 0.0:
            frost_warn = (
                f"Опасность осеннего заморозка ({day.min_temp:.1f} °C) на {day.day}!"
            )
            if frost_warn not in risks:
                risks.append(frost_warn)

        daily_delta = max(-0.5, (base_rate * temp_factor) - rain_penalty)
        # Насыщение сахаристости выше 26 Brix замедляется
        if current_brix >= 25.0:
            daily_delta = min(daily_delta, 0.08)

        current_brix = round(max(5.0, current_brix + daily_delta), 2)

        trajectory.append(
            {
                "date": day.day.isoformat(),
                "predicted_brix": current_brix,
                "mean_temp": day.mean_temp,
                "precipitation_mm": day.precipitation_mm,
            }
        )

        if current_brix >= target_brix and optimal_start is None:
            optimal_start = day.day

        if (
            optimal_start is not None
            and optimal_end is None
            and (current_brix >= target_brix + 1.5 or day.precipitation_mm >= 15.0)
        ):
            optimal_end = day.day

    if optimal_start is not None and optimal_end is None:
        optimal_end = forecast_days[-1].day if forecast_days else optimal_start

    # Формирование агрономического заключения
    if optimal_start is None:
        urgency = "low"
        recommendation = (
            f"Ягода ещё не достигла целевых кондиций ({target_brix:.1f} °Brix). "
            f"Продолжать регулярный контроль сахаристости и титруемой кислотности."
        )
    elif any("заморозка" in r for r in risks):
        urgency = "critical"
        recommendation = (
            f"КРИТИЧЕСКИ: В прогнозе заморозки! Завершить уборку урожая до {optimal_start.isoformat()} "
            f"во избежание подмерзания гроздей и потери товарного качества."
        )
    elif any("Обильные осадки" in r for r in risks):
        urgency = "high"
        recommendation = (
            f"Внимание: перед прогнозируемыми затяжными дождями рекомендуется провести сбор "
            f"в окно {optimal_start.isoformat()} — {optimal_end.isoformat() if optimal_end else ''}, "
            f"чтобы предотвратить растрескивание тонкой кожицы и вспышку Botrytis."
        )
    else:
        urgency = "medium"
        recommendation = (
            f"Благоприятные метеоусловия для уборки сорта {profile.name}. "
            f"Оптимальный диапазон сбора с {optimal_start.isoformat()} по {optimal_end.isoformat() if optimal_end else ''}."
        )

    return HarvestWindowPrediction(
        variety_name=profile.name,
        target_sugar_brix=target_brix,
        predicted_brix_trajectory=trajectory,
        optimal_window_start=optimal_start,
        optimal_window_end=optimal_end,
        harvest_urgency=urgency,
        risks_detected=risks,
        agronomic_recommendation=recommendation,
    )
