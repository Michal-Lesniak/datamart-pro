"""Feature engineering dla modelu churn.

Funkcje czyste — nie znają MLflow ani Databricks.
"""
from __future__ import annotations

from datetime import datetime
from typing import Iterable

import pandas as pd


REQUIRED_COLUMNS = {"customer_id", "total_orders", "total_spent",
                    "avg_order_value", "last_order_at", "created_at"}


def days_since(ref: datetime, ts: pd.Timestamp | None) -> float:
    """Dni między `ref` a `ts`. NULL → duża wartość (znak churna)."""
    if ts is None or pd.isna(ts):
        return 365.0
    return max((ref - ts.to_pydatetime()).total_seconds() / 86400, 0.0)


def build_feature_frame(df: pd.DataFrame, reference_date: datetime | None = None) -> pd.DataFrame:
    """Buduje feature DataFrame dla modelu churn.

    Zwraca kolumny: customer_id, recency_days, frequency, monetary, tenure_days,
    avg_order_value, aov_vs_global.
    """
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Brakujące kolumny: {sorted(missing)}")

    ref = reference_date or datetime.utcnow()

    out                        = df.copy()
    out["recency_days"]        = out["last_order_at"].apply(lambda t: days_since(ref, t))
    out["frequency"]           = out["total_orders"].fillna(0).astype(int)
    out["monetary"]            = out["total_spent"].fillna(0.0)
    out["tenure_days"]         = out["created_at"].apply(lambda t: days_since(ref, t))
    out["avg_order_value"]     = out["avg_order_value"].fillna(0.0)

    global_aov = out["avg_order_value"].mean() or 1.0
    out["aov_vs_global"]       = out["avg_order_value"] / global_aov

    return out[[
        "customer_id",
        "recency_days",
        "frequency",
        "monetary",
        "tenure_days",
        "avg_order_value",
        "aov_vs_global",
    ]]


def label_churned(df: pd.DataFrame, recency_threshold_days: int = 90) -> pd.Series:
    """Etykietuje klienta jako churn jeżeli `recency_days > threshold`."""
    if "recency_days" not in df.columns:
        raise ValueError("Wymagana kolumna `recency_days`")
    if recency_threshold_days <= 0:
        raise ValueError("`recency_threshold_days` musi być > 0")
    return (df["recency_days"] > recency_threshold_days).astype(int)


def select_feature_columns() -> list[str]:
    """Zwraca listę kolumn używanych przez model (bez `customer_id`)."""
    return [
        "recency_days",
        "frequency",
        "monetary",
        "tenure_days",
        "avg_order_value",
        "aov_vs_global",
    ]
