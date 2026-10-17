"""Модели прогноза риска (ml/pr-016, ml/pr-017) и базовые линии.

Каждая модель по обучающим годам предсказывает вероятность события на день
t + h. Базовые линии нужны, чтобы честно оценить пользу ML: «завтра как
сегодня» (persistence) и климатическая частота события в это время года."""

import copy
from dataclasses import dataclass
from typing import Any, cast

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
from torch import nn

from vitiguard_ml.risk.features import HORIZONS

SEQUENCE_DAYS = 30
# Признаки дня для LSTM: погода и состояние агромоделей
SEQUENCE_COLUMNS = (
    "mean_temp",
    "min_temp",
    "max_temp",
    "precipitation_mm",
    "mean_humidity",
    "mild_hours",
    "hours_above_35",
    "degree_days",
    "season_sin",
    "season_cos",
    "gt_index",
    "incubation",
    "infection_today",
)


def class_weight(positive: float) -> float:
    """Вес положительного класса: редкое событие весит столько, сколько
    частое; без положительных примеров (или если событие частое) — 1."""
    return (1 - positive) / positive if 0 < positive < 0.5 else 1.0


def persistence(frame: pd.DataFrame, target: str) -> np.ndarray:
    """Риск через h дней такой же, как сегодня."""
    today = frame["infection_today"] if target == "downy" else frame["gt_index"] / 100
    values: np.ndarray = today.to_numpy(dtype=float)
    return values


def climatology(
    train: pd.DataFrame, test: pd.DataFrame, column: str, window: int = 7
) -> np.ndarray:
    """Частота события в обучающих годах в те же дни года (± window дней)."""
    known = train.dropna(subset=[column])
    days = known["date"].dt.dayofyear.to_numpy()
    labels = known[column].to_numpy(dtype=float)
    rate = np.zeros(367)
    for day in range(1, 367):
        near = np.abs(days - day) <= window
        rate[day] = labels[near].mean() if near.any() else labels.mean()
    values: np.ndarray = rate[test["date"].dt.dayofyear.to_numpy()]
    return values


@dataclass
class BoostingModel:
    model: lgb.LGBMClassifier
    importance: dict[str, float]


def train_boosting(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    columns: list[str],
    column: str,
    seed: int = 42,
) -> BoostingModel:
    """LightGBM с ранней остановкой по проверочному году. Редкий класс
    (заражение милдью) уравновешивается весами."""
    positive = float(train[column].mean())
    model = lgb.LGBMClassifier(
        n_estimators=600,
        learning_rate=0.03,
        num_leaves=15,
        min_child_samples=40,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.7,
        reg_lambda=1.0,
        scale_pos_weight=class_weight(positive),
        random_state=seed,
        verbose=-1,
    )
    model.fit(
        train[columns],
        train[column].astype(int),
        eval_set=[(valid[columns], valid[column].astype(int))],
        eval_metric="average_precision",
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    gain = model.booster_.feature_importance(importance_type="gain")
    total = float(gain.sum()) or 1.0
    return BoostingModel(
        model,
        {name: float(value) / total for name, value in zip(columns, gain, strict=True)},
    )


def predict_boosting(
    model: BoostingModel, frame: pd.DataFrame, columns: list[str]
) -> np.ndarray:
    values: np.ndarray = np.asarray(model.model.predict_proba(frame[columns]))[:, 1]
    return values


class RiskLSTM(nn.Module):
    """Последовательность 30 дней → логиты события на 7 дней вперёд разом."""

    def __init__(self, features: int, hidden: int = 32) -> None:
        super().__init__()
        self.lstm = nn.LSTM(features, hidden, batch_first=True)
        self.head = nn.Linear(hidden, len(HORIZONS))

    def forward(self, sequences: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.lstm(sequences)
        logits: torch.Tensor = self.head(hidden[-1])
        return logits


def sequences(
    daily: pd.DataFrame, features: pd.DataFrame, rows: pd.Index
) -> np.ndarray:
    """Окно в SEQUENCE_DAYS дней, заканчивающееся днём строки. Индексы строк
    признаков совпадают с индексами дневной таблицы (одна точка подряд)."""
    table = daily.loc[:, list(SEQUENCE_COLUMNS)].to_numpy(dtype=np.float32)
    positions = daily.index.get_indexer(rows)
    offsets = np.arange(-SEQUENCE_DAYS + 1, 1)
    windows: np.ndarray = table[positions[:, None] + offsets[None, :]]
    return windows


@dataclass
class LSTMModel:
    model: RiskLSTM
    mean: np.ndarray
    std: np.ndarray


def _targets(frame: pd.DataFrame, target: str) -> np.ndarray:
    values: np.ndarray = frame[[f"{target}_h{h}" for h in HORIZONS]].to_numpy(
        dtype=np.float32
    )
    return values


def train_lstm(
    train_x: np.ndarray,
    train_frame: pd.DataFrame,
    valid_x: np.ndarray,
    valid_frame: pd.DataFrame,
    target: str,
    epochs: int = 25,
    seed: int = 42,
) -> LSTMModel:
    """Одна сеть на все горизонты; пустые цели (день t + h вне окна модели)
    не участвуют в потерях. Ранняя остановка по потерям проверочного года."""
    torch.manual_seed(seed)
    mean = train_x.reshape(-1, train_x.shape[-1]).mean(axis=0)
    std = train_x.reshape(-1, train_x.shape[-1]).std(axis=0) + 1e-6

    def tensors(
        x: np.ndarray, frame: pd.DataFrame
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.from_numpy((x - mean) / std),
            torch.from_numpy(_targets(frame, target)),
        )

    x_train, y_train = tensors(train_x, train_frame)
    x_valid, y_valid = tensors(valid_x, valid_frame)
    known = y_train[~torch.isnan(y_train)]
    positive = float(known.mean())
    pos_weight = torch.tensor(class_weight(positive))
    model = RiskLSTM(x_train.shape[-1])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    def loss_of(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        logits = model(x)
        mask = ~torch.isnan(y)
        losses = nn.functional.binary_cross_entropy_with_logits(
            logits[mask], y[mask], pos_weight=pos_weight
        )
        return losses

    best, best_state, patience = float("inf"), copy.deepcopy(model.state_dict()), 0
    generator = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        model.train()
        order = torch.randperm(len(x_train), generator=generator)
        for start in range(0, len(order), 256):
            batch = order[start : start + 256]
            optimizer.zero_grad()
            loss = loss_of(x_train[batch], y_train[batch])
            # В окружении без torch тип потерь — Any, с ним — без аннотаций
            cast(Any, loss).backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            valid_loss = float(loss_of(x_valid, y_valid))
        if valid_loss < best - 1e-4:
            best, best_state, patience = (
                valid_loss,
                copy.deepcopy(model.state_dict()),
                0,
            )
        else:
            patience += 1
            if patience >= 4:
                break
    model.load_state_dict(best_state)
    return LSTMModel(model, mean, std)


@torch.no_grad()
def predict_lstm(model: LSTMModel, x: np.ndarray) -> np.ndarray:
    model.model.eval()
    logits = model.model(torch.from_numpy((x - model.mean) / model.std))
    values: np.ndarray = torch.sigmoid(logits).numpy()
    return values
