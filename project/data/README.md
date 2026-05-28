# Данные проекта

Основной датасет: Calgary-HTTP из Internet Traffic Archive.

Источник: https://ita.ee.lbl.gov/html/contrib/Calgary-HTTP.html

Raw-файл:

- `data/raw/calgary_access_log.gz`

Подготовленный файл для экспериментов:

- `data/processed/calgary_http_hourly.csv`

Подготовка:

```bash
python scripts/prepare_calgary_http.py --download
```

Скрипт извлекает timestamp из HTTP-лога и агрегирует количество запросов по часам в формат:

```text
timestamp,requests
1994-10-24 13:00:00,18
...
```

Часть строк исходного лога не содержит timestamp в стандартном формате, поэтому они пропускаются при агрегации.
