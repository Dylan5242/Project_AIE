from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import DATA_PATH


def load_hourly_requests(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load prepared hourly request counts."""
    if not path.exists():
        raise FileNotFoundError(
            f"Prepared dataset is missing: {path}. "
            "Run `python scripts/prepare_calgary_http.py --download` first."
        )

    data = pd.read_csv(path, parse_dates=["timestamp"])
    data = data.sort_values("timestamp").reset_index(drop=True)
    data["requests"] = data["requests"].astype(int)
    return data


def latest_history(path: Path = DATA_PATH, history_size: int = 168) -> pd.DataFrame:
    """Return the latest history window from the prepared dataset."""
    data = load_hourly_requests(path)
    if len(data) < history_size:
        raise ValueError(f"Need at least {history_size} hourly rows, got {len(data)}.")
    return data.tail(history_size).reset_index(drop=True)


def history_ending_at_hour(
    hour: int,
    path: Path = DATA_PATH,
    history_size: int = 168,
) -> pd.DataFrame:
    """Return the latest history window ending at the requested hour."""
    if hour < 0 or hour > 23:
        raise ValueError(f"Hour must be between 0 and 23, got {hour}.")

    data = load_hourly_requests(path)
    if len(data) < history_size:
        raise ValueError(f"Need at least {history_size} hourly rows, got {len(data)}.")

    row_numbers = pd.Series(range(len(data)), index=data.index)
    candidates = data[(row_numbers >= history_size - 1) & (data["timestamp"].dt.hour == hour)]
    if candidates.empty:
        raise ValueError(f"No history window ending at hour {hour:02d}:00.")

    end_position = int(candidates.index[-1])
    start_position = end_position - history_size + 1
    return data.iloc[start_position : end_position + 1].reset_index(drop=True)
