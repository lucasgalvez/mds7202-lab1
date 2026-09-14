"""Productos analíticos Gold de Pudubella."""

from __future__ import annotations

from datetime import timedelta
from typing import Any

import polars as pl


def build_rfm_exclusions(
    orders: pl.DataFrame, payments: pl.DataFrame
) -> pl.DataFrame:
    """Registra órdenes entregadas sin pago para excluirlas de RFM."""
    delivered = orders.filter(pl.col("order_status") == "delivered")
    paid_order_ids = payments["order_id"].unique().to_list()

    sin_pago = delivered.filter(~pl.col("order_id").is_in(paid_order_ids))

    return sin_pago.select(
        ["order_id", "customer_id", "order_purchase_timestamp"]
    ).with_columns(pl.lit("delivered_order_without_payment").alias("reason"))


def build_sales_daily(
    orders: pl.DataFrame, items: pl.DataFrame
) -> pl.DataFrame:
    """Construye ventas de ítems por fecha de compra y órdenes entregadas."""
    delivered = orders.filter(pl.col("order_status") == "delivered").select(
        ["order_id", "order_purchase_timestamp"]
    )

    ventas = items.join(delivered, on="order_id", how="inner")

    return (
        ventas.with_columns(
            pl.col("order_purchase_timestamp").dt.date().alias("sale_date")
        )
        .group_by("sale_date")
        .agg(
            pl.col("price").sum().alias("items_sold_value"),
            pl.col("order_id").n_unique().alias("delivered_orders"),
        )
        .sort("sale_date")
    )


def _segment_expr(segments: dict[str, Any]) -> pl.Expr:
    """Construye la regla de segmento evaluando la configuración congelada.

    El ``for`` recorre los pocos segmentos declarados en
    ``config/rfm_segments.yaml`` (configuración), no las filas de una tabla:
    arma una expresión ``when/then`` vectorizada que Polars evalúa sobre
    todo el DataFrame de una sola vez. El primer segmento que declara la
    configuración tiene prioridad sobre los siguientes.
    """
    expr = pl.lit("Other")
    for name, rules in reversed(list(segments["segments"].items())):
        condition = pl.lit(True)
        if "max_recency_days" in rules:
            condition = condition & (
                pl.col("recency_days") <= rules["max_recency_days"]
            )
        if "min_recency_days" in rules:
            condition = condition & (
                pl.col("recency_days") >= rules["min_recency_days"]
            )
        if "min_frequency" in rules:
            condition = condition & (
                pl.col("frequency") >= rules["min_frequency"]
            )
        if "min_monetary" in rules:
            condition = condition & (
                pl.col("monetary") >= rules["min_monetary"]
            )
        expr = (
            pl.when(condition).then(pl.lit(name.capitalize())).otherwise(expr)
        )
    return expr


def build_customer_rfm(
    orders: pl.DataFrame,
    customers: pl.DataFrame,
    payments: pl.DataFrame,
    segments: dict[str, Any],
) -> pl.DataFrame:
    """Calcula RFM de compras entregadas y aplica reglas congeladas.

    Solo se consideran órdenes entregadas Y pagadas (las entregadas sin pago
    quedan en ``gold.rfm_exclusions`` y no distorsionan Monetary). El
    agrupamiento usa ``customer_unique_id`` porque un mismo cliente real
    puede tener varios ``customer_id`` (uno distinto por compra en Olist).
    """
    delivered = orders.filter(pl.col("order_status") == "delivered")
    excluded_ids = build_rfm_exclusions(orders, payments)["order_id"].to_list()
    eligible = delivered.filter(~pl.col("order_id").is_in(excluded_ids))

    payments_per_order = payments.group_by("order_id").agg(
        pl.col("payment_value").sum().alias("order_payment_value")
    )

    orders_rfm = eligible.join(
        payments_per_order, on="order_id", how="inner"
    ).join(
        customers.select(["customer_id", "customer_unique_id"]),
        on="customer_id",
        how="left",
    )

    fecha_referencia = orders_rfm[
        "order_purchase_timestamp"
    ].max().date() + timedelta(days=1)

    rfm = (
        orders_rfm.group_by("customer_unique_id")
        .agg(
            pl.col("order_purchase_timestamp").max().alias("last_purchase_at"),
            pl.col("order_id").n_unique().alias("frequency"),
            pl.col("order_payment_value").sum().alias("monetary"),
        )
        .with_columns(
            (pl.lit(fecha_referencia) - pl.col("last_purchase_at").dt.date())
            .dt.total_days()
            .alias("recency_days")
        )
    )

    return rfm.with_columns(_segment_expr(segments).alias("segment")).sort(
        "customer_unique_id"
    )
