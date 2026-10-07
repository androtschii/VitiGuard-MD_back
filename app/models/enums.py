import enum

from sqlalchemy import Enum


class ProcessingStatus(enum.StrEnum):
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"


def string_enum[E: enum.StrEnum](enum_cls: type[E], name: str) -> Enum:
    # Строка с CHECK вместо ENUM-типа PostgreSQL: новое значение добавляется без ALTER TYPE
    return Enum(
        enum_cls,
        name=name,
        native_enum=False,
        create_constraint=True,
        length=20,
        values_callable=lambda members: [member.value for member in members],
    )
