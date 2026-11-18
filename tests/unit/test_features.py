"""Testy dla `src.ml.churn_features`."""
from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
import pytest

from src.ml.churn_features import (
    build_feature_frame,
    days_since,
    label_churned,
    select_feature_columns,
)


REF = datetime(2026, 1, 1)


def _mk_df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_days_since_null_returns_default():
    assert days_since(REF, None)       == 365.0
    assert days_since(REF, pd.NaT)     == 365.0


def test_days_since_computes_correctly():
    ts = pd.Timestamp(REF - timedelta(days=10))
    assert days_since(REF, ts) == pytest.approx(10.0, rel=0.01)


def test_days_since_negative_clamps_to_zero():
    future = pd.Timestamp(REF + timedelta(days=5))
    assert days_since(REF, future) == 0.0


def test_build_feature_frame_smoke():
    df = _mk_df([
        dict(customer_id="C1", total_orders=3,  total_spent=300.0,
             avg_order_value=100.0, last_order_at=pd.Timestamp(REF - timedelta(days=30)),
             created_at=pd.Timestamp(REF - timedelta(days=500))),
    ])
    out = build_feature_frame(df, reference_date=REF)
    assert list(out.columns) == [
        "customer_id", "recency_days", "frequency", "monetary",
        "tenure_days", "avg_order_value", "aov_vs_global",
    ]
    assert out.loc[0, "recency_days"] == pytest.approx(30.0, rel=0.01)
    assert out.loc[0, "frequency"]    == 3


def test_build_feature_frame_missing_column_raises():
    df = _mk_df([dict(customer_id="C1")])
    with pytest.raises(ValueError, match="Brakujące kolumny"):
        build_feature_frame(df, reference_date=REF)


def test_build_feature_frame_handles_nulls():
    df = _mk_df([
        dict(customer_id="C1", total_orders=None, total_spent=None,
             avg_order_value=None, last_order_at=None,
             created_at=pd.Timestamp(REF - timedelta(days=500))),
    ])
    out = build_feature_frame(df, reference_date=REF)
    assert out.loc[0, "frequency"]    == 0
    assert out.loc[0, "monetary"]     == 0.0
    assert out.loc[0, "recency_days"] == 365.0


@pytest.mark.parametrize("recency_days,expected", [
    (0,    0),
    (89,   0),
    (90,   0),
    (91,   1),
    (365,  1),
])
def test_label_churned_boundary(recency_days, expected):
    df = pd.DataFrame({"recency_days": [recency_days]})
    assert label_churned(df, 90).iloc[0] == expected


@pytest.mark.parametrize("threshold", [-1, 0])
def test_label_churned_bad_threshold(threshold):
    df = pd.DataFrame({"recency_days": [10]})
    with pytest.raises(ValueError, match="> 0"):
        label_churned(df, threshold)


def test_label_churned_missing_column_raises():
    df = pd.DataFrame({"foo": [1]})
    with pytest.raises(ValueError, match="recency_days"):
        label_churned(df)


def test_select_feature_columns_stable_order():
    cols = select_feature_columns()
    assert cols[0] == "recency_days"
    assert "customer_id" not in cols
    assert len(cols) == 6
