"""Trening modelu churn + logging w MLflow."""
from __future__ import annotations

import argparse
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.ml.churn_features import (
    build_feature_frame,
    label_churned,
    select_feature_columns,
)


EXPERIMENT = "datamart-pro-churn"
MODEL_NAME = "datamart-pro-churn-model"


@dataclass(frozen=True)
class ModelSpec:
    name:   str
    estimator_cls: type
    params: dict[str, Any]


MODEL_SPECS: list[ModelSpec] = [
    ModelSpec("logreg", LogisticRegression, {"C": 1.0, "max_iter": 1000}),
    ModelSpec("rf",     RandomForestClassifier, {"n_estimators": 200, "max_depth": 8, "random_state": 42}),
    ModelSpec("gbt",    GradientBoostingClassifier, {"n_estimators": 150, "max_depth": 4, "random_state": 42}),
]


def generuj_dane_mock(n: int = 2000, seed: int = 42) -> pd.DataFrame:
    """Syntetyczne dane klientów — do uruchomienia bez Databricks."""
    rng = np.random.default_rng(seed)
    now = datetime.utcnow()
    df  = pd.DataFrame({
        "customer_id":     [f"C{i:06d}" for i in range(n)],
        "total_orders":    rng.poisson(5, n),
        "total_spent":     np.abs(rng.normal(1500, 900, n)).round(2),
        "avg_order_value": np.abs(rng.normal(300, 180, n)).round(2),
        "last_order_at":   [now - timedelta(days=int(d)) for d in rng.integers(0, 365, n)],
        "created_at":      [now - timedelta(days=int(d)) for d in rng.integers(60, 1500, n)],
    })
    return df


def _log_run(spec: ModelSpec, X_tr, X_te, y_tr, y_te, feature_names: list[str]) -> float:
    with mlflow.start_run(run_name=spec.name):
        mlflow.log_params({"model_type": spec.name, **spec.params})

        model = spec.estimator_cls(**spec.params)
        model.fit(X_tr, y_tr)

        y_pred  = model.predict(X_te)
        y_proba = model.predict_proba(X_te)[:, 1]

        metrics = {
            "roc_auc":   float(roc_auc_score(y_te, y_proba)),
            "f1":        float(f1_score(y_te, y_pred)),
            "precision": float(precision_score(y_te, y_pred, zero_division=0)),
            "recall":    float(recall_score(y_te, y_pred, zero_division=0)),
        }
        mlflow.log_metrics(metrics)

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            report = classification_report(y_te, y_pred, zero_division=0)
            (tmp / "classification_report.txt").write_text(report)

            cm = pd.DataFrame(
                confusion_matrix(y_te, y_pred),
                index=["actual_0", "actual_1"],
                columns=["pred_0", "pred_1"],
            )
            cm.to_csv(tmp / "confusion_matrix.csv")

            mlflow.log_artifact(str(tmp / "classification_report.txt"))
            mlflow.log_artifact(str(tmp / "confusion_matrix.csv"))

            if hasattr(model, "feature_importances_"):
                fi = (
                    pd.DataFrame({
                        "feature":    feature_names,
                        "importance": model.feature_importances_,
                    })
                    .sort_values("importance", ascending=False)
                )
                fi.to_csv(tmp / "feature_importances.csv", index=False)
                mlflow.log_artifact(str(tmp / "feature_importances.csv"))

            mlflow.sklearn.log_model(model, artifact_path="model")

        return metrics["roc_auc"]


def train(mock: bool = True, tracking_uri: str | None = None) -> None:
    """End-to-end trening: 3 runy + rejestracja najlepszego."""
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    mlflow.set_experiment(EXPERIMENT)

    raw       = generuj_dane_mock() if mock else _load_from_databricks()
    features  = build_feature_frame(raw)
    labels    = label_churned(features, recency_threshold_days=90)
    X_cols    = select_feature_columns()
    X         = features[X_cols].to_numpy(dtype=float)
    y         = labels.to_numpy()

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    scaler = StandardScaler().fit(X_tr)
    X_tr   = scaler.transform(X_tr)
    X_te   = scaler.transform(X_te)

    best_auc  = 0.0
    best_run  = None
    for spec in MODEL_SPECS:
        auc = _log_run(spec, X_tr, X_te, y_tr, y_te, X_cols)
        if auc > best_auc:
            best_auc = auc
            best_run = mlflow.last_active_run()

    if best_run is None:
        raise RuntimeError("Żaden run nie przeszedł — brak modelu do rejestracji")

    model_uri = f"runs:/{best_run.info.run_id}/model"
    result    = mlflow.register_model(model_uri, MODEL_NAME)

    client = mlflow.tracking.MlflowClient()
    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=result.version,
        stage="Staging",
        archive_existing_versions=True,
    )
    print(f"Zarejestrowano {MODEL_NAME} v{result.version} w stage=Staging (AUC={best_auc:.4f})")


def _load_from_databricks() -> pd.DataFrame:
    """Hook dla prawdziwych danych — kiedyś SELECT z Delta."""
    raise NotImplementedError("Uruchom z --mock lub podłącz źródło Databricks")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock",         action="store_true", help="Trenuj na danych syntetycznych")
    parser.add_argument("--tracking-uri", default=None,        help="np. ./mlruns albo databricks")
    args = parser.parse_args()
    train(mock=args.mock, tracking_uri=args.tracking_uri)


if __name__ == "__main__":
    main()
