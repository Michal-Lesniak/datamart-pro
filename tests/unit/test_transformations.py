"""Testy dla `src.pipelines.silver_transform`."""
from __future__ import annotations

from datetime import datetime

import pytest
from pyspark.sql import Row

from src.pipelines.silver_transform import (
    VALID_STATUSES,
    clean_customers,
    compute_customer_metrics,
    dedup_by_updated_at,
    enrich_customers,
    validate_customer_status,
)


def test_clean_customers_drops_null_id(spark):
    df = spark.createDataFrame([
        Row(customer_id="C1", email="A@TEST.COM", country=" pl "),
        Row(customer_id=None, email="B@TEST.COM", country=" de "),
    ])
    out = clean_customers(df).collect()
    assert len(out) == 1
    assert out[0].email   == "a@test.com"
    assert out[0].country == "PL"


def test_clean_customers_raises_without_id(spark):
    df = spark.createDataFrame([Row(email="a@b.com", country="pl")])
    with pytest.raises(ValueError, match="customer_id"):
        clean_customers(df)


def test_clean_customers_empty(spark):
    df = spark.createDataFrame([], "customer_id STRING, email STRING, country STRING")
    assert clean_customers(df).count() == 0


def test_dedup_keeps_latest(spark):
    df = spark.createDataFrame([
        Row(customer_id="C1", status="active",  updated_at=datetime(2024, 1, 1)),
        Row(customer_id="C1", status="churned", updated_at=datetime(2024, 2, 1)),
    ])
    out = dedup_by_updated_at(df).collect()
    assert len(out) == 1 and out[0].status == "churned"


def test_dedup_raises_without_updated_at(spark):
    df = spark.createDataFrame([Row(customer_id="C1")])
    with pytest.raises(ValueError, match="updated_at"):
        dedup_by_updated_at(df)


@pytest.mark.parametrize("status,expected_kept", [
    ("active",  True),
    ("churned", True),
    ("lapsed",  True),
    ("trial",   True),
    ("unknown", False),
    ("ACTIVE",  False),
    ("",        False),
])
def test_validate_customer_status(spark, status, expected_kept):
    df = spark.createDataFrame([Row(customer_id="C1", status=status)])
    cnt = validate_customer_status(df).count()
    assert (cnt == 1) == expected_kept


def test_validate_status_raises(spark):
    df = spark.createDataFrame([Row(customer_id="C1")])
    with pytest.raises(ValueError, match="status"):
        validate_customer_status(df)


def test_compute_metrics_only_paying(spark):
    df = spark.createDataFrame([
        Row(customer_id="C1", amount=100.0, status="paid",      created_at=datetime(2024, 1, 1)),
        Row(customer_id="C1", amount=50.0,  status="pending",   created_at=datetime(2024, 1, 2)),
        Row(customer_id="C1", amount=200.0, status="delivered", created_at=datetime(2024, 1, 3)),
    ])
    out = compute_customer_metrics(df).collect()
    assert len(out) == 1
    r = out[0]
    assert r.total_orders    == 2
    assert r.total_spent     == 300.0
    assert r.avg_order_value == 150.0


@pytest.mark.parametrize("cols_missing", [
    {"amount"},
    {"customer_id"},
    {"status"},
])
def test_compute_metrics_raises(spark, cols_missing):
    row = {"customer_id": "C1", "amount": 1.0, "status": "paid", "created_at": datetime(2024, 1, 1)}
    for c in cols_missing:
        row.pop(c)
    df = spark.createDataFrame([Row(**row)])
    with pytest.raises(ValueError, match="Brakujące kolumny"):
        compute_customer_metrics(df)


def test_enrich_customers_fills_nulls(spark):
    customers = spark.createDataFrame([
        Row(customer_id="C1", name="A"),
        Row(customer_id="C2", name="B"),   # brak zamówień
    ])
    metrics = spark.createDataFrame([
        Row(customer_id="C1", total_orders=3, total_spent=300.0, avg_order_value=100.0),
    ])
    out = enrich_customers(customers, metrics).orderBy("customer_id").collect()
    assert out[1].total_orders    == 0
    assert out[1].total_spent     == 0.0
    assert out[1].avg_order_value == 0.0


def test_valid_statuses_set_matches_docs():
    assert VALID_STATUSES == {"active", "churned", "lapsed", "trial"}
