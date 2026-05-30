# Self-checklist

## 1. Запуск сервиса

Status: done

Сервис запускается по инструкции из `README.md`:

```bash
uvicorn src.service.main:app --reload
```

Также есть Dockerfile.

## 2. Реальная модель в `/predict`

Status: done

Endpoint `/predict` загружает сохранённую модель:

```text
artifacts/models/random_forest_tuned.joblib
```

Код: `src/service/main.py`, `src/models/predict.py`.

## 3. EDA + эксперимент

Status: done

EDA и эксперименты находятся в:

```text
notebooks/model_experiments.ipynb
```

Есть графики временного ряда, сезонности по часам и дням недели, метрики моделей.

## 4. Сравнение моделей

Status: done

Сравнивались:

- seasonal naive;
- linear regression;
- random forest;
- histogram gradient boosting;
- PyTorch LSTM;
- tuned random forest;
- tuned histogram gradient boosting.

Метрики приведены в `README.md` и `report.md`.

## 5. Структура кода

Status: done

Код вынесен из notebook:

```text
src/models/data.py
src/models/features.py
src/models/predict.py
src/models/train.py
src/models/metrics.py
src/service/main.py
```

## 6. Развёртывание

Status: done

Есть `Dockerfile` и инструкция запуска без Docker в `README.md`.

## 7. Конфиги и секреты

Status: done

Есть `configs/.env.example`. Реальных секретов в проекте нет.

## 8. Наблюдаемость

Status: done

Есть:

- `GET /health`;
- базовое логирование запросов и ошибок в `src/service/main.py`.

## 9. Обоснование модели

Status: done

Обоснование выбора `random_forest_tuned` описано в `report.md`.

## 10. Демо-сценарий

Status: done

Демо-сценарий описан в `README.md`:

1. запустить API;
2. открыть `/docs`;
3. выполнить `/demo-request`;
4. передать JSON в `/predict`.
