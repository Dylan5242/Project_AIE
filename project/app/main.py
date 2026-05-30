from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta
from functools import lru_cache

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from src.config import DATA_PATH, HORIZON, LOG_LEVEL, MODEL_PATH
from src.data import history_ending_at_hour
from src.predict import LoadForecaster


logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("load-forecast-api")

app = FastAPI(
    title="Service Load Forecaster",
    description="Forecasts hourly service requests for the next 24 hours.",
    version="0.1.0",
)


class PredictRequest(BaseModel):
    last_timestamp: str = Field(
        ...,
        description="Timestamp of the last observed hourly request value.",
        examples=["1995-10-11T14:00:00"],
    )
    requests: list[float] = Field(
        ...,
        min_length=168,
        description="Hourly request history. The API uses the latest 168 values.",
    )


class PredictResponse(BaseModel):
    model_path: str
    last_timestamp: str
    horizon: int
    forecast: list[dict[str, float | str]]


@lru_cache(maxsize=1)
def get_forecaster() -> LoadForecaster:
    logger.info("Loading model from %s", MODEL_PATH)
    return LoadForecaster(MODEL_PATH)


def selected_demo_hour(hour: int | None) -> int:
    """Return the requested last-observation hour or the current system hour."""
    if hour is None:
        return datetime.now().hour
    return hour


def yesterday_at_hour(hour: int) -> datetime:
    yesterday = datetime.now().date() - timedelta(days=1)
    return datetime.combine(yesterday, datetime.min.time()).replace(hour=hour)


@app.get("/health")
def health() -> dict[str, object]:
    model_exists = MODEL_PATH.exists()
    data_exists = DATA_PATH.exists()
    status = "ok" if model_exists else "model_missing"
    return {
        "status": status,
        "model_path": str(MODEL_PATH),
        "model_exists": model_exists,
        "data_path": str(DATA_PATH),
        "data_exists": data_exists,
        "horizon": HORIZON,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    logger.info(
        "Received prediction request: last_timestamp=%s, history_len=%d",
        payload.last_timestamp,
        len(payload.requests),
    )
    try:
        result = get_forecaster().predict_next_24h(
            requests=payload.requests,
            last_timestamp=payload.last_timestamp,
        )
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    logger.info("Prediction completed: horizon=%d", result.horizon)
    return PredictResponse(**result.__dict__)


@app.get("/demo-request")
def demo_request(
    hour: int | None = Query(
        default=None,
        ge=0,
        le=23,
        description=(
            "Hour of the last observation in 24-hour format. "
            "Defaults to the current system hour."
        ),
        examples=[13],
    ),
) -> dict[str, object]:
    """Return a ready-to-use request body with yesterday's date."""
    last_observation_hour = selected_demo_hour(hour)
    history = history_ending_at_hour(last_observation_hour)
    return {
        "last_timestamp": yesterday_at_hour(last_observation_hour).isoformat(),
        "requests": history["requests"].astype(float).tolist(),
    }


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("APP_HOST", "0.0.0.0")
    port = int(os.getenv("APP_PORT", "8000"))
    uvicorn.run("app.main:app", host=host, port=port, reload=False)
