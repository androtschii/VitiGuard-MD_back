import enum
from typing import TYPE_CHECKING

from sqlalchemy import BigInteger, CheckConstraint, Enum, String, text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin, UUIDPrimaryKeyMixin
from app.models.enums import RiskLevel, string_enum

if TYPE_CHECKING:
    from app.models.auth import RefreshToken
    from app.models.diagnosis import LeafImage
    from app.models.vineyard import Vineyard


class UserRole(enum.StrEnum):
    ADMIN = "admin"
    AGRONOMIST = "agronomist"
    USER = "user"


class User(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "users"
    # Приложение приводит email к нижнему регистру, а база не даёт обойти это
    __table_args__ = (CheckConstraint("email = lower(email)", name="email_lowercase"),)

    email: Mapped[str] = mapped_column(String(320), unique=True)
    hashed_password: Mapped[str] = mapped_column(String(255))
    full_name: Mapped[str | None] = mapped_column(String(255))
    # Строка с CHECK вместо ENUM-типа PostgreSQL: новую роль можно добавить без ALTER TYPE
    role: Mapped[UserRole] = mapped_column(
        Enum(
            UserRole,
            name="user_role",
            native_enum=False,
            create_constraint=True,
            length=20,
            values_callable=lambda roles: [role.value for role in roles],
        ),
        server_default=UserRole.USER.value,
    )
    is_active: Mapped[bool] = mapped_column(server_default=text("true"))
    # С какого уровня риска присылать предупреждения (pr-084); None — не
    # присылать. По умолчанию — только о высоком риске
    alert_level: Mapped[RiskLevel | None] = mapped_column(
        string_enum(RiskLevel, "alert_level"), server_default=RiskLevel.HIGH.value
    )
    # Чат Telegram, привязанный к учётной записи по ссылке из профиля (pr-082):
    # из него бот принимает фото на диагностику. Один чат — одна учётная запись
    telegram_chat_id: Mapped[int | None] = mapped_column(BigInteger, unique=True)

    refresh_tokens: Mapped[list["RefreshToken"]] = relationship(
        back_populates="user", cascade="all, delete-orphan", passive_deletes=True
    )
    vineyards: Mapped[list["Vineyard"]] = relationship(
        back_populates="owner", cascade="all, delete-orphan", passive_deletes=True
    )
    leaf_images: Mapped[list["LeafImage"]] = relationship(
        back_populates="owner", cascade="all, delete-orphan", passive_deletes=True
    )
