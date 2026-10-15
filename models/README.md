# Модели

`diagnosis.onnx` — классификатор болезней по фото листа для `POST /api/v1/ai/diagnose`. Веса не хранятся в git: модель переобучается, а каждая версия весит 15 МБ и больше.

`diagnosis.json` описывает модель: название, порядок классов (порядок выходов сети) и размер входа. Он хранится в git и должен соответствовать весам.

## Как получить модель

1. Обучить модели в Google Colab ноутбуком `ml/notebooks/train_colab.ipynb` (или командой `python -m vitiguard_ml.experiments` на машине с GPU).
2. Взять из результатов `models/<модель>.onnx`, выбранную этапом `export` (сейчас `efficientnet_b0.onnx`), и положить сюда под именем `diagnosis.onnx`.
3. Проверить, что `diagnosis.json` совпадает с разделом `export` файла `results.json`.

Без модели сервер работает, а `/ai/diagnose` отвечает 503. Путь к модели задаётся переменной `VITIGUARD_DIAGNOSIS_MODEL_PATH`.
