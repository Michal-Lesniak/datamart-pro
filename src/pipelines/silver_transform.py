"""DataMart Pro — Silver layer transformations.

Czyste funkcje operujące na Spark DataFrame; testowalne w izolacji.
"""
from __future__ import annotations

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window


VALID_STATUSES = {"active", "churned", "lapsed", "trial"}


def clean_customers(df: DataFrame) -> DataFrame:
    """Czyszczenie: NULL w `customer_id` usuwane, email lowercase, country UPPER."""
    if "customer_id" not in df.columns:
        raise ValueError("Brak kolumny `customer_id`")

    return (
        df.filter(F.col("customer_id").isNotNull())
          .withColumn("email",   F.lower(F.trim(F.col("email"))))
          .withColumn("country", F.upper(F.trim(F.col("country"))))
    )


def dedup_by_updated_at(df: DataFrame, key_col: str = "customer_id") -> DataFrame:
    """Deduplikacja po kluczu — zostawia najświeższy wiersz wg `updated_at`."""
    if "updated_at" not in df.columns:
        raise ValueError("Brak kolumny `updated_at`")
    w = Window.partitionBy(key_col).orderBy(F.col("updated_at").desc_nulls_last())
    return (
        df.withColumn("_rk", F.row_number().over(w))
          .filter(F.col("_rk") == 1)
          .drop("_rk")
    )


def validate_customer_status(df: DataFrame) -> DataFrame:
    """Odrzuca wiersze ze statusem spoza VALID_STATUSES."""
    if "status" not in df.columns:
        raise ValueError("Brak kolumny `status`")
    return df.filter(F.col("status").isin(*VALID_STATUSES))


def compute_customer_metrics(orders: DataFrame) -> DataFrame:
    """Oblicza agregaty per klient: total_orders, total_spent, avg_order."""
    required = {"customer_id", "amount", "status"}
    missing = required - set(orders.columns)
    if missing:
        raise ValueError(f"Brakujące kolumny: {missing}")

    paying = orders.filter(F.col("status").isin("paid", "shipped", "delivered"))
    return (
        paying.groupBy("customer_id").agg(
            F.count("*").alias("total_orders"),
            F.sum("amount").alias("total_spent"),
            F.avg("amount").alias("avg_order_value"),
            F.max("created_at").alias("last_order_at"),
        )
    )


def enrich_customers(customers: DataFrame, metrics: DataFrame) -> DataFrame:
    """Join klientów z metrykami; NULL-safe (brak zamówień → zera)."""
    joined = customers.join(metrics, on="customer_id", how="left")
    return (
        joined
        .withColumn("total_orders",    F.coalesce(F.col("total_orders"),    F.lit(0)))
        .withColumn("total_spent",     F.coalesce(F.col("total_spent"),     F.lit(0.0)))
        .withColumn("avg_order_value", F.coalesce(F.col("avg_order_value"), F.lit(0.0)))
    )
