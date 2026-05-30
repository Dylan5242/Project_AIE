from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"

DEFAULT_DATA_PATH = DATA_DIR / "processed" / "calgary_http_hourly.csv"
DEFAULT_MODEL_PATH = MODELS_DIR / "random_forest_tuned.joblib"

MODEL_PATH = Path(os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH))
DATA_PATH = Path(os.getenv("DATA_PATH", DEFAULT_DATA_PATH))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

HISTORY_SIZE = 168
HORIZON = 24
FEATURE_COLUMNS = [
    "hour",
    "day_of_week",
    "is_weekend",
    "lag_1",
    "lag_24",
    "lag_168",
    "rolling_mean_24",
]
