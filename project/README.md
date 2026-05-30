# Итоговый проект по курсу «Инженерия Искусственного Интеллекта»

В этой папке находится итоговый мини-проект по курсу.
Проект демонстрирует применение методов и инструментов инженерии ИИ: работу с данными, эксперименты с моделями, сохранение артефактов, FastAPI-сервис для инференса и запуск через Docker.

---

## 1. Паспорт проекта

- **Название проекта:** Прогноз нагрузки на сервис
- **Автор:** `<Славенко Дмитрий Вадимович>`
- **Группа:** `<ЭФБО-06-24>`
- **Контакт:** `<slavenko056@mail.ru>`

- **Краткое описание:**  
  Проект решает задачу прогноза количества HTTP-запросов на следующие 24 часа по почасовой истории нагрузки.
  Для обучения используется открытый датасет Calgary-HTTP, агрегированный по часам.
  В экспериментах сравниваются naive baseline, линейная регрессия, random forest, histogram gradient boosting и LSTM.
  Итоговый сервис использует сохранённую модель `random_forest_tuned.joblib` и предоставляет REST API для получения прогноза.

---

## 2. Структура проекта

Проект организован в следующей структуре:

- `requirements.txt` – зависимости для локального запуска, экспериментов и notebook.
- `requirements-api.txt` – минимальные зависимости Docker-образа для API-инференса.
- `Dockerfile` – сборка Docker-образа сервиса.
- `report.md` – отчёт по проекту: постановка задачи, данные, эксперименты, результаты.
- `self-checklist.md` – чеклист самопроверки проекта перед сдачей.
- `notebooks/` – экспериментальные ноутбуки:
  - EDA;
  - сравнение моделей;
  - подбор и оценка финальной модели.
- `src/models/` – основной ML-код проекта:
  - загрузка данных;
  - генерация признаков;
  - обучение модели;
  - инференс и формирование прогноза.
- `src/service/` – FastAPI-сервис:
  - `/health`;
  - `/demo-request`;
  - `/predict`.
- `data/` – данные проекта:
  - `data/raw/` – исходный Calgary-HTTP лог;
  - `data/processed/` – почасовая агрегация запросов.
- `configs/` – конфигурационные файлы и пример `.env`.
- `tests/` – демо-скрипты и примеры запросов.
- `scripts/` – вспомогательные CLI-скрипты:
  - подготовка Calgary-HTTP;
  - подбор табличных моделей.
- `artifacts/` – сохранённые модели, таблицы результатов и графики.

---

## 3. Требования и установка

### 3.1. Требования

- Python `>= 3.12`.
- Docker Desktop, если нужен запуск через Docker.
- Git, если проект клонируется из репозитория.

### 3.2. Установка окружения

```powershell
# Перейти в папку проекта
cd project

# Создать виртуальное окружение
python -m venv .venv

# Активировать окружение на Windows
.venv\Scripts\activate

# Установить зависимости
pip install --upgrade pip
pip install -r requirements.txt
```

Для запуска только Docker-сервиса зависимости на хосте устанавливать не обязательно: Dockerfile использует `requirements-api.txt`.

---

## 4. Как запустить проект

### 4.1. Подготовка данных

В репозитории уже лежит подготовленный файл:

```text
data/processed/calgary_http_hourly.csv
```

Если нужно повторить подготовку из raw-лога:

```powershell
python scripts/prepare_calgary_http.py --download
```

### 4.2. Повторное обучение финальной модели

```powershell
python -m src.models.train
```

Финальная модель сохраняется в:

```text
artifacts/models/random_forest_tuned.joblib
```

### 4.3. Запуск сервиса локально

```powershell
uvicorn src.service.main:app --reload
```

Сервис поднимается на:

```text
http://127.0.0.1:8000
```

Swagger UI:

```text
http://127.0.0.1:8000/docs
```

### 4.4. Запуск через Docker

```powershell
docker build -t load-forecaster .
docker run --rm -p 8000:8000 load-forecaster
```

### 4.5. Проверка `/health`

```powershell
curl http://127.0.0.1:8000/health
```

Типичный ответ:

```json
{
  "status": "ok",
  "model_path": "artifacts/models/random_forest_tuned.joblib",
  "model_exists": true,
  "data_path": "data/processed/calgary_http_hourly.csv",
  "data_exists": true,
  "horizon": 24
}
```

### 4.6. Получение demo-запроса

Endpoint `/demo-request` возвращает готовое тело для `/predict`.
По умолчанию дата в `last_timestamp` берётся как вчерашний день, а час – как текущий системный час.

```powershell
curl http://127.0.0.1:8000/demo-request
```

Можно выбрать час последнего наблюдения:

```powershell
curl "http://127.0.0.1:8000/demo-request?hour=13"
```

`hour=13` означает, что последний элемент массива `requests` относится к вчерашнему дню в `13:00`.

### 4.7. Получение прогноза

Отправить прогноз можно с телом, полученным из `GET /demo-request`:

```powershell
$body = Invoke-RestMethod "http://127.0.0.1:8000/demo-request?hour=13"
Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/predict" `
  -Method Post `
  -ContentType "application/json" `
  -Body ($body | ConvertTo-Json -Depth 5)
```

Тело запроса для интеграции выглядит так.
Массив `requests` ниже сокращён для читаемости; в реальном запросе нужно передать 168 чисел.

```json
{
  "last_timestamp": "2026-05-29T13:00:00",
  "requests": [
    349,
    389,
    172
  ]
}
```

Поле `requests` содержит почасовую историю за последние 7 дней:

```text
168 = 24 часа * 7 дней
```

Последний элемент массива соответствует времени `last_timestamp`.

Можно запросить компактный прогноз только по выбранным часам:

```powershell
$body = Invoke-RestMethod "http://127.0.0.1:8000/demo-request?hour=13"
Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/predict?time_range=13:00-14:00" `
  -Method Post `
  -ContentType "application/json" `
  -Body ($body | ConvertTo-Json -Depth 5)
```

Типичный ответ:

```json
{
  "model_path": "artifacts/models/random_forest_tuned.joblib",
  "last_timestamp": "2026-05-29T13:00:00",
  "horizon": 24,
  "forecast": [
    {
      "timestamp": "2026-05-29T14:00:00",
      "requests": 299.08
    },
    {
      "timestamp": "2026-05-29T15:00:00",
      "requests": 216.53
    }
  ],
  "forecast_by_hour": {
    "13:00": 138.11,
    "14:00": 299.08
  }
}
```

Если параметр `time_range` не передан, поле `forecast_by_hour` не возвращается.
Поле `forecast` всегда содержит прогноз на 24 часа.

Краткий пример интеграции на Python:

```python
import requests

payload = requests.get("http://127.0.0.1:8000/demo-request?hour=13").json()
response = requests.post(
    "http://127.0.0.1:8000/predict?time_range=13:00-14:00",
    json=payload,
    timeout=30,
)
response.raise_for_status()
forecast = response.json()
print(forecast["forecast_by_hour"])
```

---

## 5. Данные

Для обучения используется открытый датасет Calgary-HTTP:

```text
https://ita.ee.lbl.gov/html/contrib/Calgary-HTTP.html
```

Raw-лог агрегирован по часам в файл:

```text
data/processed/calgary_http_hourly.csv
```

Формат:

```text
timestamp,requests
1994-10-24 13:00:00,33
...
```

Скрипт подготовки:

```powershell
python scripts/prepare_calgary_http.py --download
```

---

## 6. Тесты и проверки

В проекте есть демо-скрипты и ручные проверки API.

```powershell
python -m compileall src
```

```powershell
python -c "from src.service.main import app; print(app.title)"
```

```powershell
curl http://127.0.0.1:8000/health
```

Также проверялись сценарии:

- `GET /demo-request?hour=13`;
- `POST /predict`;
- `POST /predict?time_range=13:00-14:00`;
- запуск через Docker.

---

## 7. Демонстрация на защите

На защите:

1. Кратко показать структуру проекта: `notebooks/`, `src/models/`, `src/service/`, `data/`, `artifacts/`.
2. Показать notebook `notebooks/model_experiments.ipynb` с EDA, сравнением моделей и метриками.
3. Объяснить, почему финальной моделью выбрана `random_forest_tuned`.
4. Запустить сервис локально или через Docker.
5. Открыть Swagger UI:

```text
http://127.0.0.1:8000/docs
```

6. Выполнить `GET /health`.
7. Выполнить `GET /demo-request?hour=13`.
8. Скопировать полученный JSON в `POST /predict?time_range=13:00-14:00`.
9. Показать полный прогноз на 24 часа и компактный ответ `forecast_by_hour`.

---

## 8. Ограничения и дальнейшая работа

Текущие ограничения:

- модель обучена на историческом Calgary-HTTP, поэтому качество на современной реальной нагрузке может отличаться;
- сервис принимает уже агрегированный числовой ряд, но не подключается напрямую к production-логам;
- для прогноза требуется 168 почасовых значений истории;
- API-тесты можно расширить до полноценного набора `pytest`;
- крупные `.joblib` модели лучше хранить через Git LFS или внешний artifact storage.

Возможные направления развития:

- добавить мониторинг качества прогноза;
- добавить обработку новых логов и онлайн-агрегацию;
- добавить конфигурацию модели через YAML или `.env`;
- сравнить дополнительные модели временных рядов.

---

## 9. Оценка проекта

Проект ориентирован на оценку **4-5** по критериям курса:

- сервис запускается по `README.md`;
- `/predict` использует реальную сохранённую модель;
- есть данные, EDA, эксперименты и метрики;
- структура кода разделяет ML-логику и сервис;
- есть Docker-запуск;
- есть `report.md` и `self-checklist.md`.

Окончательная оценка остаётся за преподавателем и зависит от качества защиты, полноты демонстрации и выполнения чеклиста.
