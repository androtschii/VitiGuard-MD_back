import uuid
from datetime import date, datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import CheckConstraint, ForeignKey, String, UniqueConstraint, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin, UUIDPrimaryKeyMixin
from app.models.enums import Disease, RiskLevel, string_enum

if TYPE_CHECKING:
    from app.models.vineyard import Vineyard


class WeatherData(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """Почасовые метеоданные для участка из Open-Meteo."""

    __tablename__ = "weather_data"
    __table_args__ = (
        # Прогноз на час позже перезаписывается фактическими данными, а не дублируется
        UniqueConstraint("vineyard_id", "observed_at"),
        CheckConstraint("relative_humidity BETWEEN 0 AND 100", name="humidity_range"),
        CheckConstraint("precipitation_mm >= 0", name="precipitation_non_negative"),
    )

    vineyard_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("vineyards.id", ondelete="CASCADE"), index=True
    )
    observed_at: Mapped[datetime]
    temperature_c: Mapped[float]
    relative_humidity: Mapped[float]
    precipitation_mm: Mapped[float]
    is_forecast: Mapped[bool] = mapped_column(server_default=text("false"))

    vineyard: Mapped["Vineyard"] = relationship(back_populates="weather")


class DiseaseRisk(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """Риск болезни на участке на дату, рассчитанный одним методом."""

    __tablename__ = "disease_risks"
    __table_args__ = (
        # Пересчёт по расписанию обновляет строку; время пересчёта — updated_at
        UniqueConstraint("vineyard_id", "disease", "target_date", "method"),
        CheckConstraint("score BETWEEN 0 AND 1", name="score_range"),
    )

    vineyard_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("vineyards.id", ondelete="CASCADE"), index=True
    )
    disease: Mapped[Disease] = mapped_column(string_enum(Disease, "disease"))
    target_date: Mapped[date]
    # Агромодель (three_tens, goidanich, gubler_thomas) или имя ML-модели с версией
    method: Mapped[str] = mapped_column(String(100))
    score: Mapped[float]
    level: Mapped[RiskLevel] = mapped_column(string_enum(RiskLevel, "risk_level"))
    # Входные значения метода, чтобы объяснить пользователю, откуда взялся риск
    factors: Mapped[dict[str, Any] | None] = mapped_column(JSONB)

    vineyard: Mapped["Vineyard"] = relationship(back_populates="disease_risks")
