from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from .config import DEFAULT_MODEL_PATH, HISTORY_SIZE, HORIZON
from .data import load_hourly_requests


RANDOM_STATE = 42


def make_training_windows(data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Create direct 24-hour training examples for the final random forest."""
    values = data["requests"].to_numpy(dtype=np.float32)
    timestamps = data["timestamp"].reset_index(drop=True)

    x_tab, y = [], []
    for end_idx in range(HISTORY_SIZE, len(data) - HORIZON + 1):
        history = values[end_idx - HISTORY_SIZE : end_idx]
        target = values[end_idx : end_idx + HORIZON]
        origin_ts = timestamps.iloc[end_idx - 1]

        x_tab.append(
            [
                origin_ts.hour,
                origin_ts.dayofweek,
                int(origin_ts.dayofweek >= 5),
                history[-1],
                history[-24],
                history[-168],
                history[-24:].mean(),
            ]
        )
        y.append(target)

    return np.asarray(x_tab, dtype=np.float32), np.asarray(y, dtype=np.float32)


def train_final_model(output_path=DEFAULT_MODEL_PATH) -> RandomForestRegressor:
    data = load_hourly_requests()
    x_tab, y = make_training_windows(data)

    n_samples = len(y)
    train_end = int(n_samples * 0.85)

    # After model selection in the notebook, train the final model on train+validation.
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=16,
        min_samples_leaf=5,
        max_features=0.8,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(x_tab[:train_end], y[:train_end])
    joblib.dump(model, output_path)
    return model


if __name__ == "__main__":
    model = train_final_model()
    print(f"Saved final model to {DEFAULT_MODEL_PATH}")
