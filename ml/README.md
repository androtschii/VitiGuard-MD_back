# VitiGuard MD — ML

Обучение и оценка моделей: диагностика болезней по фото листьев (классификация и сегментация), прогноз риска заболеваний, поиск аномалий на спутниковых снимках. Готовые модели экспортируются в ONNX и подключаются к бэкенду.

Отдельный проект со своими зависимостями: PyTorch не попадает в образ API.

## Установка

```bash
cd ml
uv sync                 # подготовка данных и тесты, без PyTorch
uv sync --group train   # плюс PyTorch, timm, Albumentations, scikit-learn — для обучения
```

Обучение на GPU — в Google Colab или Kaggle Notebooks: клонировать репозиторий, `pip install uv`, затем те же команды.

## Структура

```
configs/           параметры экспериментов (base.yaml и наследники через extends)
src/vitiguard_ml/  код: конфиги, фиксация seed, дальше — данные, обучение, оценка
tests/             тесты
data/              наборы данных (не в git)
notebooks/         ноутбук обучения в Google Colab
artifacts/         веса, метрики, графики (не в git)
```

## Данные

```bash
uv run python -m vitiguard_ml.data.plantvillage   # PlantVillage: чёрная гниль, эска, пятнистость, здоровые
uv run python -m vitiguard_ml.data.ngld           # NGLD (Mendeley Data): милдью, оидиум, здоровые
uv run python -m vitiguard_ml.data.combine        # общий манифест data/processed/manifest.csv без повторов
uv run python -m vitiguard_ml.data.split          # splits.csv (train/val/test) и class_stats.json (веса классов)
uv run python -m vitiguard_ml.data.hermos         # полевой тестовый набор (HERMOS, архив 3,1 ГБ) — только для теста
```

Каждый набор подготавливается в `data/processed/<источник>/<класс>/` и описывается манифестом `manifest.csv`: путь, класс, источник, SHA-256, перцептивный хэш dHash, ширина, высота.

## Обучение и оценка

```bash
uv run python -m vitiguard_ml.experiments                       # все этапы
uv run python -m vitiguard_ml.experiments --stages classify --models efficientnet_b0
uv run python -m vitiguard_ml.experiments --limit 24 --epochs 1 --no-pretrained  # проверка запуска
```

Этапы: `classify` — обучение и сравнение ResNet-50, EfficientNet-B0 и ViT-B/16 на тестовой выборке и полевом наборе HERMOS; `explain` — Grad-CAM; `segment` — маски поражения OpenCV и U-Net; `export` — лучшая модель в ONNX. Результаты — в `--out` (по умолчанию `results/`): `results.json`, рисунки в `figures/`, веса в `models/`.

На GPU — ноутбук `notebooks/train_colab.ipynb` для Google Colab: архив `ml/` с `data/processed` загружается на Google Диск, ноутбук распаковывает его, ставит зависимости, запускает этапы и сохраняет результаты на Диск.

## Прогноз риска

```bash
# из корня репозитория: архив погоды, размеченный агромоделями сервера
uv run python -m app.cli.risk_dataset --out ml/data/raw/risk
# из ml/: LightGBM и LSTM против базовых линий, проверка по годам 2021-2025
uv run --group train python -m vitiguard_ml.risk.experiment
```

Результаты — `results/risk/metrics.json` и графики PR-AUC и F1 по горизонтам.

## Воспроизводимость

- все параметры эксперимента — в конфиге, опечатка в имени параметра — ошибка;
- `set_seed(config.seed)` фиксирует Python, NumPy и PyTorch, включая детерминированный cuDNN;
- порядок классов в `classes` — это порядок выходов модели.

## Проверки

```bash
uv run pytest
uv run mypy
```
