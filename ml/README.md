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
artifacts/         веса, метрики, графики (не в git)
```

## Данные

```bash
uv run python -m vitiguard_ml.data.plantvillage   # PlantVillage: чёрная гниль, эска, пятнистость, здоровые
uv run python -m vitiguard_ml.data.ngld           # NGLD (Mendeley Data): милдью, оидиум, здоровые
uv run python -m vitiguard_ml.data.combine        # общий манифест data/processed/manifest.csv без повторов
```

Каждый набор подготавливается в `data/processed/<источник>/<класс>/` и описывается манифестом `manifest.csv`: путь, класс, источник, SHA-256, перцептивный хэш dHash, ширина, высота.

## Воспроизводимость

- все параметры эксперимента — в конфиге, опечатка в имени параметра — ошибка;
- `set_seed(config.seed)` фиксирует Python, NumPy и PyTorch, включая детерминированный cuDNN;
- порядок классов в `classes` — это порядок выходов модели.

## Проверки

```bash
uv run pytest
uv run mypy
```
