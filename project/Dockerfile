FROM python:3.12-slim

WORKDIR /app

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY . .

ENV APP_HOST=0.0.0.0
ENV APP_PORT=8000
ENV TZ=Europe/Moscow
ENV MODEL_PATH=artifacts/models/random_forest_tuned.joblib
ENV DATA_PATH=data/processed/calgary_http_hourly.csv

EXPOSE 8000

CMD ["uvicorn", "src.service.main:app", "--host", "0.0.0.0", "--port", "8000"]
