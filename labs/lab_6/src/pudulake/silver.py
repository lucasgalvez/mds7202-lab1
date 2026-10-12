"""Transformaciones y reglas críticas de las entidades Silver."""

from __future__ import annotations

import polars as pl

from src.pudulake.contracts import ContractViolation

DATE_COLUMNS = (
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
)

ALLOWED_ORDER_STATUS = {
    "approved",
    "canceled",
    "created",
    "delivered",
    "invoiced",
    "processing",
    "shipped",
    "unavailable",
}


def _parse_datetime_column(
    frame: pl.DataFrame,
    column: str,
) -> pl.DataFrame:
    """Convierte una columna a Datetime distinguiendo nulos de fechas inválidas."""

    dtype = frame.schema[column]

    # Si toda la columna es nula, Polars la tipa como Null.
    if dtype == pl.Null:
        return frame.with_columns(
            pl.col(column).cast(pl.Datetime).alias(column)
        )

    # Si ya está tipada como fecha, no hacemos nada.
    if dtype == pl.Date:
        return frame.with_columns(
            pl.col(column).cast(pl.Datetime).alias(column)
        )

    if dtype.base_type() == pl.Datetime:
        return frame

    # Las fechas de origen deberían venir como String.
    if dtype != pl.String:
        raise ContractViolation(
            f"silver.orders.{column} contiene una fecha no interpretable."
        )

    parsed = pl.col(column).str.to_datetime(strict=False)

    invalid = frame.select(
        (pl.col(column).is_not_null() & parsed.is_null()).any()
    ).item()

    if invalid:
        raise ContractViolation(
            f"silver.orders.{column} contiene una fecha no interpretable."
        )

    return frame.with_columns(parsed.alias(column))


def _validate_non_negative_finite(
    frame: pl.DataFrame,
    columns: tuple[str, ...],
    table: str,
) -> None:
    """Comprueba que los montos sean no negativos y finitos."""

    for column in columns:
        values = frame[column].drop_nulls()

        if values.len() > 0 and (values < 0).any():
            raise ContractViolation(
                f"{table}.{column} contiene valores negativos."
            )

        if values.len() > 0 and (~values.is_finite()).any():
            raise ContractViolation(
                f"{table}.{column} contiene valores no finitos."
            )


def build_orders(orders: pl.DataFrame) -> pl.DataFrame:
    """Tipa fechas de órdenes y comprueba su secuencia temporal."""

    result = orders.clone()

    for column in DATE_COLUMNS:
        result = _parse_datetime_column(result, column)

    observed_status = set(
        result["order_status"].drop_nulls().unique().to_list()
    )

    invalid_status = observed_status - ALLOWED_ORDER_STATUS

    if invalid_status:
        raise ContractViolation(
            "silver.orders.order_status contiene valores fuera del contrato."
        )

    delivered_before_purchase = result.select(
        (
            (pl.col("order_status") == "delivered")
            & pl.col("order_delivered_customer_date").is_not_null()
            & (
                pl.col("order_delivered_customer_date")
                < pl.col("order_purchase_timestamp")
            )
        ).any()
    ).item()

    if delivered_before_purchase:
        raise ContractViolation(
            "silver.orders contiene una orden entregada antes de su compra."
        )

    result = result.with_columns(
        (
            (pl.col("order_status") == "delivered")
            & pl.col("order_delivered_customer_date").is_null()
        ).alias("delivery_timestamp_missing")
    )

    return result


def build_customers(customers: pl.DataFrame) -> pl.DataFrame:
    """Conserva clientes y verifica la relación uno a uno con customer_id."""

    if customers["customer_id"].null_count() > 0:
        raise ContractViolation("silver.customers.customer_id no admite nulos.")

    if customers["customer_id"].n_unique() != customers.height:
        raise ContractViolation(
            "silver.customers.customer_id debe identificar una única fila."
        )

    if customers["customer_unique_id"].null_count() > 0:
        raise ContractViolation(
            "silver.customers.customer_unique_id no admite nulos."
        )

    return customers.clone()


def build_order_items(items: pl.DataFrame) -> pl.DataFrame:
    """Comprueba que los ítems no tengan precios ni fletes negativos."""

    _validate_non_negative_finite(
        items,
        ("price", "freight_value"),
        "silver.order_items",
    )

    return items.clone()


def build_payments(payments: pl.DataFrame) -> pl.DataFrame:
    """Comprueba que los pagos no tengan montos negativos."""

    _validate_non_negative_finite(
        payments,
        ("payment_value",),
        "silver.payments",
    )

    return payments.clone()


def validate_relationships(
    orders: pl.DataFrame,
    customers: pl.DataFrame,
    items: pl.DataFrame,
    payments: pl.DataFrame,
) -> None:
    """Verifica las claves foráneas antes de construir productos Gold."""

    orphan_orders = orders.join(
        customers.select("customer_id").unique(),
        on="customer_id",
        how="anti",
    ).height

    orphan_items = items.join(
        orders.select("order_id").unique(),
        on="order_id",
        how="anti",
    ).height

    orphan_payments = payments.join(
        orders.select("order_id").unique(),
        on="order_id",
        how="anti",
    ).height

    if orphan_orders > 0:
        raise ContractViolation(
            "Se detectó una relación huérfana entre orders y customers."
        )

    if orphan_items > 0:
        raise ContractViolation(
            "Se detectó una relación huérfana entre order_items y orders."
        )

    if orphan_payments > 0:
        raise ContractViolation(
            "Se detectó una relación huérfana entre payments y orders."
        )
