import uuid
from collections.abc import Sequence

from sqlalchemy import func
from sqlalchemy.dialects.postgresql import insert

from app.integrations.open_meteo import WeatherSample
from app.models.weather import WeatherData
from app.repositories.base import Repository


class WeatherRepository(Repository[WeatherData]):
    model = WeatherData

    async def upsert(
        self,
        vineyard_id: uuid.UUID,
        samples: Sequence[WeatherSample],
        *,
        is_forecast: bool = False,
    ) -> int:
        """Записывает часы погоды; уже записанный час обновляется, а не дублируется.

        Так повторный импорт того же периода безопасен, а фактические данные
        заменяют прогноз на тот же час."""
        if not samples:
            return 0
        rows = [
            {
                "vineyard_id": vineyard_id,
                "observed_at": sample.observed_at,
                "temperature_c": sample.temperature_c,
                "relative_humidity": sample.relative_humidity,
                "precipitation_mm": sample.precipitation_mm,
                "is_forecast": is_forecast,
            }
            for sample in samples
        ]
        statement = insert(WeatherData).values(rows)
        statement = statement.on_conflict_do_update(
            index_elements=["vineyard_id", "observed_at"],
            set_={
                "temperature_c": statement.excluded.temperature_c,
                "relative_humidity": statement.excluded.relative_humidity,
                "precipitation_mm": statement.excluded.precipitation_mm,
                "is_forecast": statement.excluded.is_forecast,
                "updated_at": func.now(),
            },
        )
        await self.session.execute(statement)
        return len(rows)
