import os
from pathlib import Path

import mlflow

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models"
TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').as_posix()}",
)
MODEL_NAME = "iris-classifier"
MODEL_ALIAS = "champion"


def configure_tracking() -> None:
    mlflow.set_tracking_uri(TRACKING_URI)
