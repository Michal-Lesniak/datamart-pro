"""Testy dla `src.ml.churn_predict`."""
from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from src.ml import churn_predict
from src.ml.churn_predict import MODEL_NAME, load_model, predict_batch


REF = datetime(2026, 1, 1)


def _mk_customers(n: int) -> pd.DataFrame:
    return pd.DataFrame({
        "customer_id":     [f"C{i}" for i in range(n)],
        "total_orders":    [5] * n,
        "total_spent":     [1000.0] * n,
        "avg_order_value": [200.0] * n,
        "last_order_at":   [pd.Timestamp(REF - timedelta(days=30))] * n,
        "created_at":      [pd.Timestamp(REF - timedelta(days=500))] * n,
    })


@pytest.mark.parametrize("stage", ["", "Development", "staging", "production", "PROD", "None"])
def test_load_model_invalid_stage_raises(stage):
    with pytest.raises(ValueError, match="Niedozwolony stage"):
        load_model(stage=stage)


@pytest.mark.parametrize("stage", ["Staging", "Production"])
def test_load_model_valid_stage_builds_correct_uri(monkeypatch, stage):
    captured: dict = {}

    def fake_load(uri: str):
        captured["uri"] = uri
        return MagicMock(name="pyfunc_model")

    monkeypatch.setattr("mlflow.pyfunc.load_model", fake_load)

    model = load_model(stage=stage)
    assert model is not None
    assert captured["uri"] == f"models:/{MODEL_NAME}/{stage}"


def test_load_model_default_stage_is_production(monkeypatch):
    captured: dict = {}
    monkeypatch.setattr(
        "mlflow.pyfunc.load_model",
        lambda uri: captured.setdefault("uri", uri) or MagicMock(),
    )
    load_model()
    assert captured["uri"].endswith("/Production")


def test_predict_batch_empty_returns_empty_frame():
    out = predict_batch(pd.DataFrame())
    assert list(out.columns) == ["customer_id", "churn_probability"]
    assert out.empty


def test_predict_batch_returns_predictions(monkeypatch):
    fake_model = MagicMock()
    fake_model.predict.return_value = np.array([0.1, 0.9, 0.5])
    monkeypatch.setattr(
        "src.ml.churn_predict.load_model",
        lambda stage="Production": fake_model,
    )

    out = predict_batch(_mk_customers(3))

    assert list(out.columns) == ["customer_id", "churn_probability"]
    assert len(out) == 3
    assert list(out["customer_id"]) == ["C0", "C1", "C2"]
    np.testing.assert_allclose(out["churn_probability"], [0.1, 0.9, 0.5])
    fake_model.predict.assert_called_once()


def test_predict_batch_passes_only_feature_columns(monkeypatch):
    captured: dict = {}

    def fake_predict(X):
        captured["cols"] = list(X.columns)
        return np.zeros(len(X))

    fake_model = MagicMock()
    fake_model.predict.side_effect = fake_predict
    monkeypatch.setattr(
        "src.ml.churn_predict.load_model",
        lambda stage="Production": fake_model,
    )

    predict_batch(_mk_customers(2))

    assert "customer_id" not in captured["cols"]
    assert "recency_days" in captured["cols"]
    assert "monetary" in captured["cols"]


def test_predict_batch_default_stage_is_production(monkeypatch):
    captured: dict = {}

    def fake_load(stage: str = "Production"):
        captured["stage"] = stage
        m = MagicMock()
        m.predict.return_value = np.zeros(1)
        return m

    monkeypatch.setattr("src.ml.churn_predict.load_model", fake_load)

    predict_batch(_mk_customers(1))
    assert captured["stage"] == "Production"


@pytest.mark.parametrize("stage", ["Staging", "Production"])
def test_predict_batch_passes_stage_through(monkeypatch, stage):
    captured: dict = {}

    def fake_load(stage: str = "Production"):
        captured["stage"] = stage
        m = MagicMock()
        m.predict.return_value = np.zeros(1)
        return m

    monkeypatch.setattr("src.ml.churn_predict.load_model", fake_load)

    predict_batch(_mk_customers(1), stage=stage)
    assert captured["stage"] == stage


def test_model_name_constant_exposed():
    assert churn_predict.MODEL_NAME == "datamart-pro-churn-model"
