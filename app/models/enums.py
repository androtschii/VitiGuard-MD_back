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


class Disease(enum.StrEnum):
    """Болезнь винограда, для которой считается риск."""

    DOWNY_MILDEW = "downy_mildew"
    POWDERY_MILDEW = "powdery_mildew"


class RiskLevel(enum.StrEnum):
    """Уровень риска заражения: низкий, средний или высокий."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Fungicide(enum.StrEnum):
    """Тип препарата по действию — от него зависит, сколько держится защита."""

    # Контактный (медь, манкоцеб): защищает до дождя, смывается ливнем
    CONTACT = "contact"
    # Системный: проникает в ткани, лечит в первые дни после заражения
    SYSTEMIC = "systemic"
    # Сера — против оидиума; в жару обжигает листья
    SULFUR = "sulfur"
