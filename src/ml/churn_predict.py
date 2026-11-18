"""Batch-scoring churn — czyta features i zapisuje score do Delta."""
from __future__ import annotations

import logging
from typing import Any

import mlflow
import mlflow.pyfunc
import pandas as pd

from src.ml.churn_features import build_feature_frame, select_feature_columns

log        = logging.getLogger("datamart.predict")
MODEL_NAME = "datamart-pro-churn-model"


def load_model(stage: str = "Production") -> Any:
    """Ładuje najnowszy model ze wskazanego stage."""
    if stage not in {"Staging", "Production"}:
        raise ValueError(f"Niedozwolony stage: {stage}")
    return mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/{stage}")


def predict_batch(customers_df: pd.DataFrame, stage: str = "Production") -> pd.DataFrame:
    """Dla każdego klienta zwraca `churn_probability`."""
    if customers_df.empty:
        return pd.DataFrame(columns=["customer_id", "churn_probability"])

    features = build_feature_frame(customers_df)
    X        = features[select_feature_columns()]
    model    = load_model(stage=stage)
    proba    = model.predict(X)

    return pd.DataFrame({
        "customer_id":       features["customer_id"].values,
        "churn_probability": proba,
    })
