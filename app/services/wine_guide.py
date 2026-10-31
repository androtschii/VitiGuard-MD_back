"""Подсказки бота: чем обрабатывать лозу, какое вино выбрать и где его купить.

Ответы готовые, без внешней языковой модели. Точки виноделен и магазинов взяты
из OpenStreetMap. Препаратов с дозами здесь нет: их выбирают по Госреестру."""

import json
from dataclasses import dataclass
from functools import lru_cache
from math import asin, cos, radians, sin, sqrt
from pathlib import Path

WINERIES_FILE = Path(__file__).resolve().parent.parent / "data" / "wineries.json"
EARTH_KM = 6371

SPRAY_TEXT = (
    "Чем обрабатывать виноград — общие подходы:\n\n"
    "• Милдью: контактные препараты, например медные, до дождя; системные — "
    "в первые дни после заражения.\n"
    "• Оидиум: препараты на основе серы или системные фунгициды. В сильную жару "
    "серой не работают: возможен ожог.\n"
    "• Серая гниль: прореживание листвы у гроздей и ботрициды перед смыканием "
    "гроздей и в начале созревания.\n"
    "• Гроздевая листовёртка: срок показывают феромонные ловушки; биопрепараты "
    "на Bacillus thuringiensis или разрешённые инсектициды — к отрождению гусениц.\n"
    "• Паутинный клещ: акарициды только при заметном заселении, лишние обработки "
    "уничтожают хищных клещей.\n"
    "• Филлоксера: опрыскивание не решает проблему, основная защита — устойчивые "
    "подвои.\n\n"
    "Препарат и дозу выбирайте по Госреестру разрешённых средств и этикетке, "
    "срок ожидания записывайте в журнал. Не уверены, что на лозе, — пришлите фото "
    "листа. При сомнениях спросите агронома."
)

WINE_TEXT = (
    "Какое вино выбрать — сочетания с молдавскими сортами:\n\n"
    "• Красное мясо и гриль: Fetească Neagră, Rară Neagră, Cabernet Sauvignon.\n"
    "• Рыба и салаты: Fetească Albă, Viorica, Sauvignon Blanc.\n"
    "• Сыры и брынза: Fetească Regală, сухое розовое.\n"
    "• Десерты: сладкий мускат, ледяное вино.\n"
    "• Праздник: игристое, например из Криково.\n\n"
    "Где купить — команда /buy. Продажа алкоголя только совершеннолетним."
)

BUY_TEXT = (
    "Отправьте свою точку кнопкой ниже: покажу ближайшие винодельни и винные "
    "магазины. Можно посмотреть и на карте в приложении: слой «Винодельни и "
    "магазины»."
)

NOTHING_NEAR = "Рядом с этой точкой в моём списке ничего нет."

LOCATION_KEYBOARD = {
    "keyboard": [[{"text": "Отправить мою точку", "request_location": True}]],
    "resize_keyboard": True,
    "one_time_keyboard": True,
}

SPRAY_WORDS = (
    "брызг",
    "опрыск",
    "вредител",
    "паразит",
    "милдью",
    "оидиум",
    "клещ",
    "листовёрт",
    "листоверт",
    "гниль",
    "филлоксер",
)
BUY_WORDS = ("купить", "покупк", "магазин", "винодел", "винари")
WINE_WORDS = ("вино", "вина", "вину")


@dataclass(frozen=True)
class Winery:
    name: str
    kind: str
    lat: float
    lon: float
    website: str | None
    phone: str | None
    approximate: bool


@lru_cache(maxsize=1)
def load_wineries() -> tuple[Winery, ...]:
    data = json.loads(WINERIES_FILE.read_text(encoding="utf-8"))
    return tuple(
        Winery(
            name=item["properties"]["name"],
            kind=item["properties"]["kind"],
            lon=item["geometry"]["coordinates"][0],
            lat=item["geometry"]["coordinates"][1],
            website=item["properties"].get("website"),
            phone=item["properties"].get("phone"),
            approximate=item["properties"].get("approximate", False),
        )
        for item in data["features"]
    )


def distance_km(lat: float, lon: float, winery: Winery) -> float:
    d_lat = radians(winery.lat - lat)
    d_lon = radians(winery.lon - lon)
    a = (
        sin(d_lat / 2) ** 2
        + cos(radians(lat)) * cos(radians(winery.lat)) * sin(d_lon / 2) ** 2
    )
    return 2 * EARTH_KM * asin(sqrt(a))


def nearest_text(lat: float, lon: float, count: int = 5, max_km: float = 150) -> str:
    ranked = sorted(
        ((distance_km(lat, lon, winery), winery) for winery in load_wineries()),
        key=lambda pair: pair[0],
    )
    near = [pair for pair in ranked[:count] if pair[0] <= max_km]
    if not near:
        return NOTHING_NEAR
    lines = ["Ближайшие к вам:"]
    for km, winery in near:
        kind = "винодельня" if winery.kind == "winery" else "магазин"
        lines.append(f"\n• {winery.name} — {kind}, {km:.0f} км")
        if winery.approximate:
            lines.append("  Точный адрес не подтверждён, показан город")
        if winery.phone:
            lines.append(f"  {winery.phone}")
        if winery.website:
            lines.append(f"  {winery.website}")
    lines.append("\nЧасы работы и наличие уточняйте у продавца. Данные OpenStreetMap.")
    return "\n".join(lines)


def guide_reply(text: str) -> str | None:
    """Ответ на вопрос своими словами; покупка важнее выбора вина."""
    question = text.lower()
    if any(word in question for word in BUY_WORDS):
        return BUY_TEXT
    if any(word in question for word in SPRAY_WORDS):
        return SPRAY_TEXT
    if any(word in question for word in WINE_WORDS):
        return WINE_TEXT
    return None
