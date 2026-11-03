from datetime import datetime

from pydantic import Field

from app.schemas.base import RequestSchema, ResponseSchema


class RobotTelemetryCreate(RequestSchema):
    """Телеметрия бортового компьютера робота-сборщика."""

    robot_id: str = Field(..., examples=["harvester-01"])
    battery_pct: float = Field(ge=0, le=100, examples=[84.5])
    speed_mps: float = Field(ge=0, le=10, default=0.5)
    current_row: str = Field(default="row_01")
    lat: float | None = Field(default=None, ge=-90, le=90)
    lon: float | None = Field(default=None, ge=-180, le=180)
    camera_fps: float = Field(ge=0, default=30.0)
    infer_latency_ms: float = Field(ge=0, default=24.5)
    active_model_version: str = Field(default="v001")
    harvested_bunches: int = Field(ge=0, default=0)
    discarded_bunches: int = Field(ge=0, default=0)


class RobotTelemetryRead(ResponseSchema):
    """Ответ сервера на телеметрию робота: статус и сигнал обновления модели."""

    robot_id: str
    status: str = "ok"
    received_at: datetime
    recommended_speed_mps: float = 0.5
    active_model_version: str
    update_available: bool = False
    latest_model_version: str | None = None


class HardSampleUploadResult(ResponseSchema):
    """Результат загрузки сложного кадра активного обучения с робота."""

    sample_id: str
    status: str = "queued_for_label_studio"
    filename: str
    bytes_received: int
