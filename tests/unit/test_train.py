"""Testy dla `src.ml.churn_train`."""
from __future__ import annotations

import dataclasses
import sys

import mlflow
import pandas as pd
import pytest

from src.ml import churn_train
from src.ml.churn_train import (
    MODEL_NAME,
    MODEL_SPECS,
    ModelSpec,
    _load_from_databricks,
    generuj_dane_mock,
    main,
    train,
)


@pytest.fixture(autouse=True)
def _mlflow_isolation():
    """Każdy test dostaje świeży stan — koniec wiszącego runu + reset URI."""
    original_uri = mlflow.get_tracking_uri()
    yield
    while mlflow.active_run():
        mlflow.end_run()
    mlflow.set_tracking_uri(original_uri)


# --- generuj_dane_mock ------------------------------------------------------


def test_generuj_dane_mock_default_shape_and_columns():
    df = generuj_dane_mock()
    assert len(df) == 2000
    expected = {"customer_id", "total_orders", "total_spent",
                "avg_order_value", "last_order_at", "created_at"}
    assert expected.issubset(df.columns)


@pytest.mark.parametrize("n", [1, 10, 500, 1234])
def test_generuj_dane_mock_respects_n(n):
    assert len(generuj_dane_mock(n=n)) == n


def test_generuj_dane_mock_is_deterministic_for_same_seed():
    df1 = generuj_dane_mock(n=50, seed=123)
    df2 = generuj_dane_mock(n=50, seed=123)
    seed_driven = ["customer_id", "total_orders", "total_spent", "avg_order_value"]
    pd.testing.assert_frame_equal(df1[seed_driven], df2[seed_driven])


def test_generuj_dane_mock_differs_for_different_seeds():
    df1 = generuj_dane_mock(n=50, seed=1)
    df2 = generuj_dane_mock(n=50, seed=2)
    assert not df1["total_spent"].equals(df2["total_spent"])


def test_generuj_dane_mock_customer_ids_are_unique():
    df = generuj_dane_mock(n=100)
    assert df["customer_id"].nunique() == 100


def test_generuj_dane_mock_monetary_fields_non_negative():
    df = generuj_dane_mock(n=500)
    assert (df["total_spent"] >= 0).all()
    assert (df["avg_order_value"] >= 0).all()
    assert (df["total_orders"] >= 0).all()


# --- ModelSpec / MODEL_SPECS ------------------------------------------------


def test_model_specs_has_three_distinct_models():
    assert len(MODEL_SPECS) == 3
    assert {s.name for s in MODEL_SPECS} == {"logreg", "rf", "gbt"}


def test_model_spec_is_frozen():
    spec = MODEL_SPECS[0]
    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.name = "other"


def test_model_spec_params_match_estimators():
    for spec in MODEL_SPECS:
        # Must be instantiable with the declared params.
        est = spec.estimator_cls(**spec.params)
        assert est is not None


# --- _load_from_databricks --------------------------------------------------


def test_load_from_databricks_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="--mock"):
        _load_from_databricks()


def test_train_without_mock_raises(tmp_path):
    with pytest.raises(NotImplementedError):
        train(mock=False, tracking_uri=str(tmp_path / "mlruns"))


# --- main / CLI -------------------------------------------------------------


def test_main_forwards_cli_flags(monkeypatch):
    captured: dict = {}

    def fake_train(mock, tracking_uri):
        captured["mock"] = mock
        captured["tracking_uri"] = tracking_uri

    monkeypatch.setattr("src.ml.churn_train.train", fake_train)
    monkeypatch.setattr(sys, "argv", ["churn_train", "--mock", "--tracking-uri", "./foo"])

    main()
    assert captured == {"mock": True, "tracking_uri": "./foo"}


def test_main_defaults(monkeypatch):
    captured: dict = {}
    monkeypatch.setattr(
        "src.ml.churn_train.train",
        lambda mock, tracking_uri: captured.update(mock=mock, tracking_uri=tracking_uri),
    )
    monkeypatch.setattr(sys, "argv", ["churn_train"])

    main()
    assert captured["mock"] is False
    assert captured["tracking_uri"] is None


# --- end-to-end mini-train --------------------------------------------------


def test_train_end_to_end_registers_model_in_staging(tmp_path, monkeypatch):
    """Mały trening z file-based MLflow — weryfikuje całą ścieżkę rejestracji."""
    tiny = generuj_dane_mock(n=120, seed=7)
    monkeypatch.setattr(
        "src.ml.churn_train.generuj_dane_mock",
        lambda n=2000, seed=42: tiny,
    )

    fast_spec = ModelSpec("logreg", MODEL_SPECS[0].estimator_cls, {"C": 1.0, "max_iter": 100})
    monkeypatch.setattr("src.ml.churn_train.MODEL_SPECS", [fast_spec])

    tracking_uri = str(tmp_path / "mlruns")
    train(mock=True, tracking_uri=tracking_uri)

    client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    assert versions, "Brak zarejestrowanych wersji modelu"
    assert any(v.current_stage == "Staging" for v in versions)

    experiment = client.get_experiment_by_name(churn_train.EXPERIMENT)
    assert experiment is not None
    runs = client.search_runs([experiment.experiment_id])
    assert len(runs) >= 1
    for metric in ("roc_auc", "f1", "precision", "recall"):
        assert metric in runs[0].data.metrics
