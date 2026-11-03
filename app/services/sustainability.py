"""Сервис расчёта углеродного баланса (OIV-CST 503-2016) и экологического аудита виноградника (sprint-31).

Реализует:
1. Моделирование депонирования углерода многолетней биомассой лозы (ствол, кордоны, многолетние корни).
2. Оценку секвестрации углерода почвой при различных режимах залужения междурядий (постоянные травосмеси, бобовые).
3. Расчёт прямых эмиссий дизельных тракторов (Scope 1) в сравнении с электрическими автономными роботами.
4. Расчёт прямых и косвенных выбросов закиси азота (N2O) от минеральных удобрений по методологии IPCC Tier 1 (GWP 273).
5. Расчёт нетто-баланса терруара (Net Carbon Sink) и потенциальных углеродных кредитов (€/год).
6. Аудит соответствия европейскому и молдавскому органическому стандарту (EU 2018/848: лимит меди 4.0 кг/га, запрет гербицидов).
"""

from app.schemas.sustainability import (
    CarbonBalanceRequest,
    CarbonBalanceResponse,
    CarbonBreakdownDTO,
    OrganicAuditRequest,
    OrganicAuditResponse,
    RegionalCarbonBenchmarkDTO,
    VineyardPracticesDTO,
)

# Коэффициент эмиссии дизельного топлива (IPCC / DEFRA): 2.68 кг CO2-экв / литр
DIESEL_EMISSION_FACTOR_KG_PER_L = 2.68

# Углеродоёмкость энергосистемы Молдовы/Румынии: 0.38 кг CO2-экв / кВт·ч
GRID_ELECTRIC_FACTOR_KG_PER_KWH = 0.38

# Потенциал глобального потепления закиси азота N2O (IPCC AR6, 100-летний горизонт)
N2O_GWP = 273.0

# Молярные массы: N2O (44) / N2 (28) = 1.571; CO2 (44) / C (12) = 3.667
CO2_TO_C_RATIO = 44.0 / 12.0
N2O_TO_N_RATIO = 44.0 / 28.0

# Максимально допустимая норма меди в органическом виноделии (EU 2018/848): 4.0 кг Cu/га/год
ORGANIC_COPPER_LIMIT_KG_HA = 4.0


def calculate_woody_sequestration(vines_per_ha: int, vine_age_years: int) -> float:
    """Депонирование углерода в постоянную многолетнюю древесину лозы (т CO2-экв / га / год).

    Молодой виноградник (<3 лет) аккумулирует меньше, зрелый (4-25 лет) стабильно ~0.35 кг сухой массы/куст.
    """
    age_factor = min(1.0, max(0.2, vine_age_years / 5.0))
    annual_dry_biomass_kg = 0.35 * age_factor
    carbon_fraction = 0.48  # доля углерода в абсолютно сухой древесине лозы

    c_stored_kg = vines_per_ha * annual_dry_biomass_kg * carbon_fraction
    co2e_tons = (c_stored_kg * CO2_TO_C_RATIO) / 1000.0
    return round(co2e_tons, 3)


def calculate_cover_crop_sequestration(cover_crop_type: str) -> float:
    """Накопление органического углерода в почве (SOC) за счёт корневой массы трав (т CO2-экв / га / год)."""
    mapping = {
        "bare_soil": 0.0,
        "spontaneous": 0.45,
        "permanent_grass": 0.95,
        "legumes": 1.40,
    }
    return mapping.get(cover_crop_type, 0.50)


def calculate_pruning_retention(residue_management: str) -> float:
    """Удержание углерода при обращении с обрезками лозы (т CO2-экв / га / год)."""
    mapping = {
        "burned": -0.20,  # Эмиссия дыма и продуктов неполного сгорания
        "removed": 0.0,
        "mulched_in_situ": 0.55,  # Измельчение и заделка в почву
        "biochar": 1.25,  # Пиролиз в биоуголь длительного депонирования
    }
    return mapping.get(residue_management, 0.0)


def calculate_n2o_fertilizer_emissions(kg_n_ha: float) -> float:
    """Прямые выбросы N2O от внесения азотных удобрений по методологии IPCC Tier 1 (т CO2-экв / га / год).

    Формула: N_applied * 0.01 (эмиссионный фактор 1%) * (44/28) * GWP(273) / 1000
    """
    direct_n2o_n = kg_n_ha * 0.01
    n2o_kg = direct_n2o_n * N2O_TO_N_RATIO
    co2e_kg = n2o_kg * N2O_GWP
    return round(co2e_kg / 1000.0, 3)


def calculate_carbon_breakdown(practices: VineyardPracticesDTO) -> CarbonBreakdownDTO:
    """Рассчитывает полную структуру поглощения и выбросов парниковых газов на 1 га."""
    # 1. Поглощение (Sequestration)
    wood_seq = calculate_woody_sequestration(
        practices.vines_per_ha, practices.vine_age_years
    )
    cover_seq = calculate_cover_crop_sequestration(practices.inter_row_cover_crop)
    pruning_seq = calculate_pruning_retention(practices.pruning_residue_management)
    total_seq = max(0.0, wood_seq + cover_seq + pruning_seq)

    # 2. Выбросы (Emissions)
    effective_diesel_l = practices.diesel_tractor_liters_per_ha
    electric_emissions = 0.0

    if practices.electric_robot_operations:
        # Электрические роботы-комбайны и опрыскиватели сокращают расход солярки на 65%
        effective_diesel_l = practices.diesel_tractor_liters_per_ha * 0.35
        # Энергопотребление роботов ~140 кВт·ч/га/год
        electric_emissions = (140.0 * GRID_ELECTRIC_FACTOR_KG_PER_KWH) / 1000.0

    diesel_emissions = (effective_diesel_l * DIESEL_EMISSION_FACTOR_KG_PER_L) / 1000.0
    n2o_emissions = calculate_n2o_fertilizer_emissions(
        practices.nitrogen_fertilizer_kg_n_ha
    )
    total_emissions = diesel_emissions + electric_emissions + n2o_emissions

    # 3. Нетто-баланс (положительный = поглотитель, отрицательный = источник)
    net_balance = total_seq - total_emissions

    return CarbonBreakdownDTO(
        vine_woody_biomass_sequestration=wood_seq,
        soil_cover_crop_sequestration=cover_seq,
        pruning_mulch_retention=pruning_seq,
        total_sequestration_tco2e_ha=round(total_seq, 3),
        diesel_tractor_emissions_tco2e_ha=round(diesel_emissions, 3),
        electric_equipment_emissions_tco2e_ha=round(electric_emissions, 3),
        nitrogen_n2o_emissions_tco2e_ha=n2o_emissions,
        total_emissions_tco2e_ha=round(total_emissions, 3),
        net_carbon_balance_tco2e_ha=round(net_balance, 3),
    )


def compute_carbon_balance(request: CarbonBalanceRequest) -> CarbonBalanceResponse:
    """Генерирует официальный паспорт углеродного баланса виноградника."""
    breakdown = calculate_carbon_breakdown(request.practices)
    farm_net = breakdown.net_carbon_balance_tco2e_ha * request.practices.area_ha
    is_sink = breakdown.net_carbon_balance_tco2e_ha > 0.0

    # Потенциальные углеродные офсетные квоты (если хозяйство является нетто-поглотителем)
    credits_eur = 0.0
    if is_sink:
        credits_eur = farm_net * request.carbon_credit_price_eur_ton

    diesel_reduction_pct = 65.0 if request.practices.electric_robot_operations else 0.0

    return CarbonBalanceResponse(
        vineyard_id=request.vineyard_id,
        area_ha=request.practices.area_ha,
        breakdown_per_ha=breakdown,
        total_farm_net_tco2e_year=round(farm_net, 2),
        is_carbon_sink=is_sink,
        electric_robot_diesel_reduction_pct=diesel_reduction_pct,
        potential_carbon_credits_eur=round(credits_eur, 2),
    )


def audit_organic_practices(request: OrganicAuditRequest) -> OrganicAuditResponse:
    """Проверяет соблюдение регламентов органического и регенеративного виноградарства."""
    p = request.practices
    score = 100.0
    deficiencies: list[str] = []
    recommendations: list[str] = []

    # 1. Проверка лимита меди (4.0 кг/га/год)
    if p.copper_applied_kg_ha > ORGANIC_COPPER_LIMIT_KG_HA:
        copper_status = "exceeded"
        score -= 40.0
        deficiencies.append(
            f"Превышен лимит меди: {p.copper_applied_kg_ha} кг Cu/га при максимуме {ORGANIC_COPPER_LIMIT_KG_HA} кг/га по EU 2018/848."
        )
        recommendations.append(
            "Заменить часть медьсодержащих фунгицидов на биопрепараты (Bacillus subtilis, хитозан, сера)."
        )
    elif p.copper_applied_kg_ha >= 3.2:
        copper_status = "warning"
        score -= 10.0
        recommendations.append(
            "Внесение меди близко к предельному нормативу (80%). Рекомендуется оптимизация дозировок через VRA."
        )
    else:
        copper_status = "ok"

    # 2. Проверка химических гербицидов
    if p.synthetic_herbicides_used:
        score -= 35.0
        deficiencies.append(
            "Применение синтетических гербицидов (глифосат и аналоги) категорически запрещено в органике."
        )
        recommendations.append(
            "Перейти на механическую культивацию приствольной полосы или роботизированную прополку."
        )

    # 3. Проверка биоразнообразия и залужения
    if p.inter_row_cover_crop == "bare_soil":
        score -= 15.0
        deficiencies.append(
            "Черный пар в междурядьях приводит к водной эрозии почвы на склонах и потере углерода."
        )
        recommendations.append(
            "Засеять междурядья многолетними засухоустойчивыми злаками или вико-овсяной смесью."
        )
        cover_status = "poor"
    else:
        cover_status = "optimal"

    # 4. Проверка утилизации лозы
    if p.pruning_residue_management == "burned":
        score -= 10.0
        deficiencies.append(
            "Сжигание лозы приводит к выбросам углекислого газа и потере органического вещества."
        )
        recommendations.append(
            "Использовать тракторный измельчитель-мульчер для возврата биомассы в почву."
        )

    final_score = max(0.0, score)
    ready = (
        (final_score >= 80.0)
        and (copper_status != "exceeded")
        and (not p.synthetic_herbicides_used)
    )

    return OrganicAuditResponse(
        vineyard_id=request.vineyard_id,
        compliance_score_pct=round(final_score, 1),
        is_organic_certified_ready=ready,
        copper_limit_status=copper_status,
        copper_applied_kg_ha=p.copper_applied_kg_ha,
        herbicide_free=not p.synthetic_herbicides_used,
        biodiversity_cover_crop_status=cover_status,
        deficiencies=deficiencies,
        recommendations=recommendations,
    )


def get_regional_benchmarks() -> list[RegionalCarbonBenchmarkDTO]:
    """Возвращает эталонные региональные бенчмарки углеродного следа виноградарства."""
    return [
        RegionalCarbonBenchmarkDTO(
            region_name="Пуркарь (Юго-Восточная Молдова)",
            conventional_net_tco2e_ha=-0.45,  # традиционный: небольшой чистый источник CO2
            regenerative_net_tco2e_ha=1.85,  # с травами и роботами: сильный поглотитель
            mean_diesel_l_ha=110.0,
            recommended_cover_crop="Festuca ovina + Lotus corniculatus",
        ),
        RegionalCarbonBenchmarkDTO(
            region_name="Кодру (Центральная Молдова)",
            conventional_net_tco2e_ha=-0.30,
            regenerative_net_tco2e_ha=2.10,
            mean_diesel_l_ha=105.0,
            recommended_cover_crop="Lolium perenne + Trifolium repens",
        ),
        RegionalCarbonBenchmarkDTO(
            region_name="Бордо (Франция, Эталон OIV)",
            conventional_net_tco2e_ha=-0.80,
            regenerative_net_tco2e_ha=1.60,
            mean_diesel_l_ha=135.0,
            recommended_cover_crop="Vicia sativa + Avena sativa",
        ),
    ]
