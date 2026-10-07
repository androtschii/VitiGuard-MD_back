#!/bin/sh
# Подменяет одноимённый скрипт образа postgis/postgis. Тот при первом запуске
# ставит в базу postgis, postgis_topology и postgis_tiger_geocoder, а расширения
# у нас включают миграции.
