# Прогноз нагрузки на сервис

ML-проект для прогноза количества HTTP-запросов на следующие 24 часа по почасовой истории нагрузки.

## Данные

ИДля обучения использовался Calgary-HTTP:

https://ita.ee.lbl.gov/html/contrib/Calgary-HTTP.html

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

## Модели

В notebook сравнивались:

- seasonal naive `lag_24`;
- seasonal naive `lag_168`;
- linear regression;
- random forest;
- histogram gradient boosting;
- PyTorch LSTM;
- tuned random forest;
- tuned histogram gradient boosting.

Финальная модель для API:

```text
artifacts/models/random_forest_tuned.joblib
```

Она выбрана по лучшим метрикам на validation и test.

## Метрики на test

| model | MAE | RMSE | sMAPE |
|---|---:|---:|---:|
| random_forest_tuned | 45.02 | 67.03 | 46.49 |
| random_forest | 45.84 | 68.23 | 47.16 |
| hist_gradient_boosting_tuned | 46.67 | 69.46 | 47.43 |
| hist_gradient_boosting | 48.73 | 72.41 | 49.43 |
| pytorch_lstm | 49.52 | 75.44 | 51.39 |
| linear_regression | 58.19 | 82.67 | 58.27 |
| seasonal_naive_lag_24 | 60.75 | 89.52 | 61.95 |
| seasonal_naive_lag_168 | 61.30 | 91.77 | 63.22 |

## Структура проекта

Проект сохранён в структуре шаблона репозитория:

- `notebooks/` — EDA и эксперименты;
- `src/` — загрузка данных, признаки, обучение и inference;
- `app/` — FastAPI-приложение;
- `data/` — raw и processed данные;
- `configs/` — шаблон переменных окружения;
- `tests/` — тестовые/демо-скрипты;
- `artifacts/` — сохранённые модели и таблицы результатов;
- `report.md` — отчёт;
- `self-checklist.md` — самооценка.

## Локальный запуск

Из папки `project/`:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Проверка сервиса:

```bash
curl http://127.0.0.1:8000/health
```
(должен быть ответ "ok")

Получить готовый пример тела запроса:

```bash
curl http://127.0.0.1:8000/demo-request
```

Отправить прогноз:

```bash
curl -X POST http://127.0.0.1:8000/predict ^
  -H "Content-Type: application/json" ^
  -d "{\"last_timestamp\":\"1995-10-11T14:00:00\",\"requests\":[168 hourly values]}"
```

Для простоты ручной демонстрации лучше открыть `http://127.0.0.1:8000/docs`, выполнить `GET /demo-request`, затем скопировать JSON в `POST /predict`.

## Docker

Из папки `project/`:

```bash
docker build -t load-forecaster .
docker run --rm -p 8000:8000 load-forecaster
```

После запуска:

```bash
curl http://127.0.0.1:8000/health
```

## Повторная подготовка данных (если нужно)

```bash
python scripts/prepare_calgary_http.py --download
```

## Повторное обучение финальной модели

```bash
python -m src.train
```

Эксперименты и подбор гиперпараметров находятся в:

```text
notebooks/model_experiments.ipynb
```
