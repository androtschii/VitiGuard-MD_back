from pathlib import Path

import pytest
from pydantic import ValidationError

from vitiguard_ml.config import load_config

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def test_base_config_is_valid() -> None:
    config = load_config(CONFIGS / "base.yaml")

    assert config.seed == 42
    assert config.classes[0] == "healthy"
    assert len(config.classes) == 6
    assert config.split.train + config.split.val + config.split.test == pytest.approx(1)


def test_experiment_extends_base(tmp_path: Path) -> None:
    (tmp_path / "base.yaml").write_text(
        (CONFIGS / "base.yaml").read_text(encoding="utf-8"), encoding="utf-8"
    )
    experiment = tmp_path / "resnet.yaml"
    experiment.write_text(
        "extends: base.yaml\nname: resnet50\nepochs: 5\nsplit: {train: 0.8, val: 0.1, test: 0.1}\n",
        encoding="utf-8",
    )

    config = load_config(experiment)

    assert config.name == "resnet50"
    assert config.epochs == 5
    assert config.split.train == 0.8
    # Не переопределённое берётся из base.yaml
    assert config.image_size == 224


@pytest.mark.parametrize(
    "override",
    [
        "split: {train: 0.5, val: 0.1, test: 0.1}",
        "classes: [healthy, healthy]",
        "learning_rate: 0",
        "epohcs: 10",
    ],
    ids=["split-sum", "duplicate-class", "zero-lr", "typo"],
)
def test_invalid_config_is_rejected(tmp_path: Path, override: str) -> None:
    (tmp_path / "base.yaml").write_text(
        (CONFIGS / "base.yaml").read_text(encoding="utf-8"), encoding="utf-8"
    )
    broken = tmp_path / "broken.yaml"
    broken.write_text(f"extends: base.yaml\n{override}\n", encoding="utf-8")

    with pytest.raises(ValidationError):
        load_config(broken)
