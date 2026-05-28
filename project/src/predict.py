from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .config import DEFAULT_MODEL_PATH, HORIZON
from .features import make_feature_vector, make_future_timestamps


@dataclass
class ForecastResult:
    model_path: str
    last_timestamp: str
    horizon: int
    forecast: list[dict[str, Any]]


class LoadForecaster:
    """Wrapper around a saved sklearn direct multi-output forecaster."""

    def __init__(self, model_path: Path = DEFAULT_MODEL_PATH):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file is missing: {self.model_path}")
        self.model = joblib.load(self.model_path)

    def predict_next_24h(self, requests: list[float], last_timestamp: str | pd.Timestamp) -> ForecastResult:
        timestamp = pd.Timestamp(last_timestamp)
        feature_vector = make_feature_vector(requests, timestamp)
        prediction = np.asarray(self.model.predict(feature_vector)[0], dtype=float)
        prediction = np.clip(prediction, 0, None)
        future_timestamps = make_future_timestamps(timestamp, horizon=HORIZON)

        forecast = [
            {
                "timestamp": ts.isoformat(),
                "requests": round(float(value), 2),
            }
            for ts, value in zip(future_timestamps, prediction)
        ]

        return ForecastResult(
            model_path=str(self.model_path),
            last_timestamp=timestamp.isoformat(),
            horizon=HORIZON,
            forecast=forecast,
        )
