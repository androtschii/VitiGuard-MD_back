# Модели

Модели для `POST /api/v1/ai/diagnose`. Веса не хранятся в git: модели переобучаются, а каждая версия весит от 15 до 100 МБ.

| Файл | Что делает | Обязателен |
|---|---|---|
| `diagnosis.onnx` | классификатор болезней (EfficientNet-B0) с картой активации классов для тепловой карты | да: без него диагностика отвечает 503 |
| `lesion.onnx` | сегментация поражения (U-Net, энкодер ResNet-34) | нет: без него диагноз ставится без маски и степени поражения |

Рядом с каждой моделью лежит `.json` с тем же именем: название модели, размер входа и (для классификатора) порядок классов — порядок выходов сети. Описания хранятся в git и должны соответствовать весам.

## Как получить модели

1. Обучить модели в Google Colab ноутбуком `ml/notebooks/train_colab.ipynb` (или командой `python -m vitiguard_ml.experiments` на машине с GPU) и скачать веса из `results/models/` (`efficientnet_b0.pt`, `unet.pt`).
2. Экспортировать их для сервера из каталога `ml/`:

   ```bash
   uv sync --group train
   uv run python -m vitiguard_ml.export_server --weights <папка с весами> --out ../models
   ```

   Скрипт записывает `diagnosis.onnx`, `lesion.onnx` и их описания.

Пути к моделям задаются переменными `VITIGUARD_DIAGNOSIS_MODEL_PATH` и `VITIGUARD_LESION_MODEL_PATH`.
