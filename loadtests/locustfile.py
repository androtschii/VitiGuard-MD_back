"""Нагрузочный тест API (pr-096): так ведёт себя кабинет виноградаря.

    uv run python -m app.cli.load_seed --users 50
    VITIGUARD_RATE_LIMIT_ENABLED=false uv run uvicorn app.main:app --workers 4
    uv run --group load locust -f loadtests/locustfile.py --host http://127.0.0.1:8000

Каждый виртуальный пользователь входит один раз под своей учётной записью из
пула load_seed и дальше открывает главную, карту и участки. Запросы к
внешним сервисам (погода Open-Meteo, пересчёт риска, вероятность вспышки)
сюда не входят: меряется свой сервер, а не чужой. Ограничение частоты на
время прогона выключают — иначе все пользователи с одного адреса упрутся в
предел входа."""

import itertools
import math
import os
import random

from locust import HttpUser, between, task

API = "/api/v1"
PASSWORD = "load-test-123"
USERS = int(os.environ.get("LOAD_USERS", "50"))
# Учётные записи раздаются по кругу: при 200 пользователях Locust на пул в
# 50 записей по одной записи работают четверо
numbers = itertools.cycle(range(1, USERS + 1))


def tile(lat: float, lon: float, zoom: int) -> str:
    n = 2**zoom
    x = int((lon + 180) / 360 * n)
    y = int((1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n)
    return f"{zoom}/{x}/{y}"


class Grower(HttpUser):
    # Человек читает страницу несколько секунд, прежде чем перейти дальше
    wait_time = between(1, 5)

    def on_start(self) -> None:
        email = f"load-{next(numbers):03d}@load.vitiguard.md"
        response = self.client.post(
            f"{API}/auth/login",
            json={"email": email, "password": PASSWORD},
            name="/auth/login",
        )
        token = response.json()["access_token"]
        self.client.headers["Authorization"] = f"Bearer {token}"
        vineyards = self.client.get(f"{API}/vineyards", name="/vineyards").json()
        self.vineyards = [item["id"] for item in vineyards["items"]]

    def vineyard(self) -> str:
        return random.choice(self.vineyards)

    @task(5)
    def home(self) -> None:
        # Главная: профиль, участки, уровни риска, колокольчик
        self.client.get(f"{API}/users/me", name="/users/me")
        self.client.get(f"{API}/vineyards", name="/vineyards")
        self.client.get(f"{API}/risks/today", name="/risks/today")
        self.client.get(f"{API}/alerts/unread-count", name="/alerts/unread-count")

    @task(3)
    def map_view(self) -> None:
        lat, lon = 47.14 + random.uniform(-0.1, 0.1), 28.86 + random.uniform(-0.1, 0.1)
        bbox = f"{lon - 0.02:.4f},{lat - 0.01:.4f},{lon + 0.02:.4f},{lat + 0.01:.4f}"
        self.client.get(
            f"{API}/osm/vineyards", params={"bbox": bbox}, name="/osm/vineyards"
        )
        self.client.get(
            f"{API}/tiles/ndvi/{tile(lat, lon, 15)}.png", name="/tiles/ndvi"
        )

    @task(3)
    def vineyard_card(self) -> None:
        vineyard = self.vineyard()
        self.client.get(f"{API}/vineyards/{vineyard}", name="/vineyards/{id}")
        self.client.get(
            f"{API}/vineyards/{vineyard}/risks", name="/vineyards/{id}/risks"
        )
        self.client.get(
            f"{API}/vineyards/{vineyard}/vegetation", name="/vineyards/{id}/vegetation"
        )
        self.client.get(
            f"{API}/vineyards/{vineyard}/treatments/plan",
            name="/vineyards/{id}/treatments/plan",
        )

    @task(1)
    def history(self) -> None:
        self.client.get(f"{API}/diagnoses", name="/diagnoses")
        self.client.get(f"{API}/alerts", name="/alerts")

    @task(1)
    def report(self) -> None:
        # Тяжёлые ответы: вёрстка PDF и книга XLSX
        vineyard = self.vineyard()
        self.client.get(
            f"{API}/vineyards/{vineyard}/report.pdf", name="/vineyards/{id}/report.pdf"
        )
        self.client.get(f"{API}/vineyards/{vineyard}/data", name="/vineyards/{id}/data")
