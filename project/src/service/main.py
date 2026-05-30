from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta
from functools import lru_cache

from fastapi import FastAPI, HTTPException, Query
import pandas as pd
from pydantic import BaseModel, Field

from src.models.config import DATA_PATH, HORIZON, LOG_LEVEL, MODEL_PATH
from src.models.data import history_ending_at_hour
from src.models.predict import LoadForecaster


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
    forecast_by_hour: dict[str, float] | None = None


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


def parse_time_range(time_range: str) -> list[str]:
    """Parse an inclusive full-hour range like 13:00-14:00."""
    try:
        raw_start, raw_end = time_range.split("-", maxsplit=1)
        start = datetime.strptime(raw_start.strip(), "%H:%M").time()
        end = datetime.strptime(raw_end.strip(), "%H:%M").time()
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail="time_range must use HH:MM-HH:MM format, for example 13:00-14:00.",
        ) from exc

    if start.minute != 0 or end.minute != 0:
        raise HTTPException(
            status_code=400,
            detail="Only full-hour ranges are supported, for example 13:00-14:00.",
        )

    hours: list[str] = []
    current_hour = start.hour
    while True:
        hours.append(f"{current_hour:02d}:00")
        if current_hour == end.hour:
            return hours
        current_hour = (current_hour + 1) % 24


def forecast_by_hour(forecast: list[dict[str, float | str]], hours: list[str]) -> dict[str, float]:
    values_by_hour = {
        pd.Timestamp(item["timestamp"]).strftime("%H:%M"): float(item["requests"])
        for item in forecast
    }
    return {hour: values_by_hour[hour] for hour in hours if hour in values_by_hour}


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


@app.post("/predict", response_model=PredictResponse, response_model_exclude_none=True)
def predict(
    payload: PredictRequest,
    time_range: str | None = Query(
        default=None,
        description=(
            "Optional inclusive full-hour range for a compact forecast view, "
            "for example 13:00-14:00."
        ),
        examples=["13:00-14:00"],
    ),
) -> PredictResponse:
    selected_hours = parse_time_range(time_range) if time_range is not None else None
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
    response = PredictResponse(**result.__dict__)
    if selected_hours is not None:
        response.forecast_by_hour = forecast_by_hour(response.forecast, selected_hours)
    return response


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
    uvicorn.run("src.service.main:app", host=host, port=port, reload=False)
