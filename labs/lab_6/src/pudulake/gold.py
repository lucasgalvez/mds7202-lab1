"""Productos analíticos Gold de Pudubella."""

from __future__ import annotations

from datetime import timedelta
from typing import Any

import polars as pl


def build_rfm_exclusions(
    orders: pl.DataFrame,
    payments: pl.DataFrame,
) -> pl.DataFrame:
    """Registra órdenes entregadas sin pago para excluirlas de RFM."""

    paid_orders = payments.select("order_id").unique()

    return (
        orders.filter(pl.col("order_status") == "delivered")
        .join(
            paid_orders,
            on="order_id",
            how="anti",
        )
        .select(
            "order_id",
            "customer_id",
            "order_purchase_timestamp",
        )
        .with_columns(pl.lit("delivered_order_without_payment").alias("reason"))
        .sort("order_id")
    )


def build_sales_daily(
    orders: pl.DataFrame,
    items: pl.DataFrame,
) -> pl.DataFrame:
    """Construye ventas por fecha."""

    return (
        orders.filter(pl.col("order_status") == "delivered")
        .select(
            "order_id",
            "order_purchase_timestamp",
        )
        .join(
            items.select(
                "order_id",
                "price",
            ),
            on="order_id",
            how="inner",
        )
        .with_columns(
            pl.col("order_purchase_timestamp").dt.date().alias("sale_date")
        )
        .group_by("sale_date")
        .agg(
            pl.col("price").sum().alias("items_sold_value"),
            pl.col("order_id").n_unique().alias("delivered_orders"),
        )
        .sort("sale_date")
    )


def build_customer_rfm(
    orders: pl.DataFrame,
    customers: pl.DataFrame,
    payments: pl.DataFrame,
    segments: dict[str, Any],
) -> pl.DataFrame:
    """Calcula RFM por persona."""

    payments_by_order = payments.group_by("order_id").agg(
        pl.col("payment_value").sum().alias("order_payment_value")
    )

    eligible = (
        orders.filter(pl.col("order_status") == "delivered")
        .select(
            "order_id",
            "customer_id",
            "order_purchase_timestamp",
        )
        .join(
            payments_by_order,
            on="order_id",
            how="inner",
        )
        .join(
            customers.select(
                "customer_id",
                "customer_unique_id",
            ),
            on="customer_id",
            how="inner",
        )
    )

    if eligible.is_empty():
        return pl.DataFrame(
            schema={
                "customer_unique_id": pl.String,
                "last_purchase": pl.Datetime,
                "frequency": pl.UInt32,
                "monetary": pl.Float64,
                "recency_days": pl.Int64,
                "segment": pl.String,
            }
        )

    latest_purchase = eligible["order_purchase_timestamp"].max()

    reference_date = latest_purchase.date() + timedelta(days=1)

    rfm = (
        eligible.group_by("customer_unique_id")
        .agg(
            pl.col("order_purchase_timestamp").max().alias("last_purchase"),
            pl.col("order_id").n_unique().alias("frequency"),
            pl.col("order_payment_value").sum().alias("monetary"),
        )
        .with_columns(
            (pl.lit(reference_date) - pl.col("last_purchase").dt.date())
            .dt.total_days()
            .alias("recency_days")
        )
    )

    rules = segments["segments"]

    champions = rules["champions"]
    loyal = rules["loyal"]
    new = rules["new"]
    lost = rules["lost"]

    return (
        rfm.with_columns(
            pl.when(
                (pl.col("recency_days") <= champions["max_recency_days"])
                & (pl.col("frequency") >= champions["min_frequency"])
                & (pl.col("monetary") >= champions["min_monetary"])
            )
            .then(pl.lit("Champions"))
            .when(pl.col("frequency") >= loyal["min_frequency"])
            .then(pl.lit("Loyal"))
            .when(pl.col("recency_days") <= new["max_recency_days"])
            .then(pl.lit("New"))
            .when(pl.col("recency_days") >= lost["min_recency_days"])
            .then(pl.lit("Lost"))
            .otherwise(pl.lit("Regular"))
            .alias("segment")
        )
        .select(
            "customer_unique_id",
            "last_purchase",
            "frequency",
            "monetary",
            "recency_days",
            "segment",
        )
        .sort("customer_unique_id")
    )
