from __future__ import annotations

import numpy as np
import pandas as pd

from .config import FEATURE_COLUMNS, HISTORY_SIZE


def make_feature_vector(requests: list[float], last_timestamp: pd.Timestamp) -> np.ndarray:
    """Build one tabular feature row for a 24-hour direct forecaster."""
    if len(requests) < HISTORY_SIZE:
        raise ValueError(f"Need at least {HISTORY_SIZE} hourly request values.")

    history = np.asarray(requests[-HISTORY_SIZE:], dtype=np.float32)
    last_timestamp = pd.Timestamp(last_timestamp)

    features = [
        last_timestamp.hour,
        last_timestamp.dayofweek,
        int(last_timestamp.dayofweek >= 5),
        history[-1],
        history[-24],
        history[-168],
        float(history[-24:].mean()),
    ]
    return np.asarray(features, dtype=np.float32).reshape(1, -1)


def build_feature_frame(requests: list[float], last_timestamp: pd.Timestamp) -> pd.DataFrame:
    """Return features as a named DataFrame for easier debugging."""
    vector = make_feature_vector(requests, last_timestamp)
    return pd.DataFrame(vector, columns=FEATURE_COLUMNS)


def make_future_timestamps(last_timestamp: pd.Timestamp, horizon: int = 24) -> list[pd.Timestamp]:
    """Build hourly timestamps for the forecast horizon."""
    start = pd.Timestamp(last_timestamp) + pd.Timedelta(hours=1)
    return list(pd.date_range(start, periods=horizon, freq="h"))
