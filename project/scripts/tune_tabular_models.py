from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "calgary_http_hourly.csv"
MODELS_DIR = PROJECT_ROOT / "models"
HISTORY_SIZE = 168
HORIZON = 24
RANDOM_STATE = 42


def make_forecast_windows(data: pd.DataFrame):
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


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def smape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    return float(100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + eps)))


def evaluate(model_name: str, params: dict, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "model": model_name,
        "params": params,
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "sMAPE": smape(y_true, y_pred),
    }


def main() -> None:
    df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
    x_tab, y = make_forecast_windows(df)

    n_samples = len(y)
    train_end = int(n_samples * 0.70)
    val_end = int(n_samples * 0.85)

    x_train, x_val, x_test = x_tab[:train_end], x_tab[train_end:val_end], x_tab[val_end:]
    y_train, y_val, y_test = y[:train_end], y[train_end:val_end], y[val_end:]

    rf_grid = [
        {"n_estimators": 200, "max_depth": 14, "min_samples_leaf": 3, "max_features": 1.0},
        {"n_estimators": 300, "max_depth": 16, "min_samples_leaf": 3, "max_features": 1.0},
        {"n_estimators": 300, "max_depth": 18, "min_samples_leaf": 5, "max_features": 1.0},
        {"n_estimators": 300, "max_depth": None, "min_samples_leaf": 5, "max_features": 1.0},
        {"n_estimators": 300, "max_depth": 16, "min_samples_leaf": 5, "max_features": 0.8},
        {"n_estimators": 400, "max_depth": 12, "min_samples_leaf": 2, "max_features": 1.0},
    ]

    boosting_grid = [
        {"max_iter": 250, "learning_rate": 0.05, "max_leaf_nodes": 31, "l2_regularization": 0.01},
        {"max_iter": 200, "learning_rate": 0.04, "max_leaf_nodes": 31, "l2_regularization": 0.05},
        {"max_iter": 300, "learning_rate": 0.03, "max_leaf_nodes": 31, "l2_regularization": 0.05},
        {"max_iter": 200, "learning_rate": 0.05, "max_leaf_nodes": 15, "l2_regularization": 0.05},
        {"max_iter": 300, "learning_rate": 0.03, "max_leaf_nodes": 15, "l2_regularization": 0.10},
        {"max_iter": 200, "learning_rate": 0.08, "max_leaf_nodes": 15, "l2_regularization": 0.10},
    ]

    results = []
    best_models = {}

    for params in rf_grid:
        print(f"Training random_forest {params}")
        model = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1, **params)
        model.fit(x_train, y_train)
        val_pred = model.predict(x_val)
        row = evaluate("random_forest_tuned", params, y_val, val_pred)
        print(row)
        results.append(row)
        if "random_forest_tuned" not in best_models or row["MAE"] < best_models["random_forest_tuned"][0]["MAE"]:
            best_models["random_forest_tuned"] = (row, model)

    for params in boosting_grid:
        print(f"Training hist_gradient_boosting {params}")
        base_model = HistGradientBoostingRegressor(random_state=RANDOM_STATE, **params)
        model = MultiOutputRegressor(base_model, n_jobs=1)
        model.fit(x_train, y_train)
        val_pred = model.predict(x_val)
        row = evaluate("hist_gradient_boosting_tuned", params, y_val, val_pred)
        print(row)
        results.append(row)
        if "hist_gradient_boosting_tuned" not in best_models or row["MAE"] < best_models["hist_gradient_boosting_tuned"][0]["MAE"]:
            best_models["hist_gradient_boosting_tuned"] = (row, model)

    results_df = pd.DataFrame(results).sort_values("MAE")
    print("\nValidation ranking")
    print(results_df.to_string(index=False))

    for model_name, (row, model) in best_models.items():
        path = MODELS_DIR / f"{model_name}.joblib"
        joblib.dump(model, path)
        test_pred = model.predict(x_test)
        test_metrics = evaluate(model_name, row["params"], y_test, test_pred)
        print(f"\nSaved {model_name}: {path}")
        print("Best validation:", row)
        print("Test:", test_metrics)


if __name__ == "__main__":
    main()
