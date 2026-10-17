import csv
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest

from app.cli import risk_dataset
from app.core.errors import ExternalServiceError
from app.integrations.open_meteo import WeatherSample
from app.services.agromodels import LOCAL_TZ, HourlyWeather

POINT = risk_dataset.Point("cricova", "codru", 47.14, 28.86)


def season_hours(rain_day: date) -> list[WeatherSample]:
    """Март — июнь при 18 °C, дождь 12 мм в rain_day."""
    start = datetime(2025, 3, 1, tzinfo=UTC) - timedelta(hours=2)
    samples = []
    for hour in range(24 * 122):
        moment = start + timedelta(hours=hour)
        wet = moment.astimezone(LOCAL_TZ).date() == rain_day
        samples.append(
            WeatherSample(
                observed_at=moment,
                temperature_c=18.0,
                relative_humidity=70.0,
                precipitation_mm=0.5 if wet else 0.0,
            )
        )
    return samples


def as_hours(samples: list[WeatherSample]) -> list[HourlyWeather]:
    return [
        HourlyWeather(
            s.observed_at, s.temperature_c, s.relative_humidity, s.precipitation_mm
        )
        for s in samples
    ]


def test_rows_carry_weather_and_labels_inside_season_windows() -> None:
    rows = risk_dataset.label_rows(POINT, as_hours(season_hours(date(2025, 5, 20))))
    by_day = {row["date"]: row for row in rows}

    march = by_day["2025-03-15"]
    assert march["point"] == "cricova"
    assert march["region"] == "codru"
    assert march["mean_temp"] == 18
    # До апреля модели не применяются — разметка пустая
    assert march["three_tens"] == ""
    assert march["gubler_thomas"] == ""
    assert by_day["2025-05-20"]["three_tens"] == 1
    assert by_day["2025-05-20"]["precipitation_mm"] == pytest.approx(12)
    assert float(str(by_day["2025-05-22"]["goidanich_progress"])) > 0
    assert by_day["2025-05-22"]["symptoms_expected"] == 0
    assert by_day["2025-04-10"]["gubler_thomas"] == 0
    assert float(str(by_day["2025-06-28"]["degree_days"])) > float(
        str(by_day["2025-04-28"]["degree_days"])
    )


class FakeClient:
    def __init__(self, failures: int = 0) -> None:
        self.failures = failures
        self.calls = 0

    async def history(
        self, _lat: float, _lon: float, _start: date, _end: date
    ) -> list[WeatherSample]:
        self.calls += 1
        if self.calls <= self.failures:
            raise ExternalServiceError("Сервис погоды недоступен")
        return season_hours(date(2025, 5, 20))


async def test_collect_retries_after_refusal() -> None:
    client = FakeClient(failures=1)

    rows = await risk_dataset.collect(
        client,  # type: ignore[arg-type]
        (POINT,),
        date(2025, 3, 1),
        date(2025, 6, 30),
    )

    assert client.calls == 2
    assert {row["point"] for row in rows} == {"cricova"}


async def test_collect_gives_up_after_attempts() -> None:
    with pytest.raises(ExternalServiceError):
        await risk_dataset.history_with_retry(
            FakeClient(failures=5),  # type: ignore[arg-type]
            POINT,
            date(2025, 3, 1),
            date(2025, 6, 30),
            pause=0,
        )


def test_main_writes_csv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_collect(*_: object) -> list[dict[str, object]]:
        return risk_dataset.label_rows(POINT, as_hours(season_hours(date(2025, 5, 20))))

    monkeypatch.setattr(risk_dataset, "collect", fake_collect)

    risk_dataset.main(["--out", str(tmp_path / "risk")])

    with (tmp_path / "risk" / "daily.csv").open(encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert tuple(rows[0]) == risk_dataset.COLUMNS
    # С 1 марта по 30 июня
    assert len(rows) == 122
