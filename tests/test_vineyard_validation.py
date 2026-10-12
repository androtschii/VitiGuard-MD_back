import pytest

from app.services.vineyard import _reason_message


@pytest.mark.parametrize(
    ("reason", "message"),
    [
        ("Self-intersection[28.861 47.1405]", "Контур пересекает сам себя"),
        ("Ring Self-intersection[28.86 47.14]", "Контур пересекает сам себя"),
        ("Hole lies outside shell[28.9 47.2]", "Вырез лежит за пределами участка"),
        ("Something new[1 2]", "Контур некорректен: Something new[1 2]"),
    ],
)
def test_reasons_are_explained_in_russian(reason: str, message: str) -> None:
    assert _reason_message(reason) == message
