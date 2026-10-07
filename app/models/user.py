import enum
from typing import TYPE_CHECKING

from sqlalchemy import Enum, String, text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin, UUIDPrimaryKeyMixin

if TYPE_CHECKING:
    from app.models.auth import RefreshToken


class UserRole(enum.StrEnum):
    ADMIN = "admin"
    AGRONOMIST = "agronomist"
    USER = "user"


class User(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "users"

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

    refresh_tokens: Mapped[list["RefreshToken"]] = relationship(
        back_populates="user", cascade="all, delete-orphan", passive_deletes=True
    )
