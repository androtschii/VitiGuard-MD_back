# PostgreSQL 18 с PostGIS для сервера (pr-100). Образ postgis/postgis
# выпускается только для amd64: здесь официальный postgres с пакетом PostGIS из
# apt.postgresql.org — он есть и для amd64, и для ARM
FROM postgres:18

RUN apt-get update     && apt-get install -y --no-install-recommends postgresql-18-postgis-3     && rm -rf /var/lib/apt/lists/*
