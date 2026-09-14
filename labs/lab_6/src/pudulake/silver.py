"""Transformaciones y reglas críticas de las entidades Silver."""

from __future__ import annotations

import polars as pl

from .contracts import ContractViolation

# Todas las columnas de fecha de `orders` llegan desde Bronze como texto.
ORDER_DATE_COLUMNS = (
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
)


def _parse_dates(frame: pl.DataFrame, columns: tuple[str, ...]) -> pl.DataFrame:
    """Tipa columnas de texto a datetime distinguiendo nulo real de texto roto.

    Un valor nulo representa una fecha que legítimamente no ha ocurrido
    todavía (por ejemplo, una orden que aún no se entrega). Un valor no nulo
    que no puede interpretarse como fecha es un dato corrupto y debe detener
    la corrida antes de llegar a Gold.
    """
    result = frame
    for column in columns:
        parsed = result[column].cast(pl.String).str.to_datetime(strict=False)
        invalid = result.filter(pl.col(column).is_not_null() & parsed.is_null())
        if invalid.height > 0:
            raise ContractViolation(
                f"silver.orders.{column} contiene una fecha no interpretable."
            )
        result = result.with_columns(parsed.alias(column))
    return result


def build_orders(orders: pl.DataFrame) -> pl.DataFrame:
    """Tipa fechas de órdenes y comprueba su secuencia temporal.

    Agrega ``delivery_timestamp_missing`` para distinguir, después de tipar,
    las órdenes que nunca registraron una entrega (nulo real) de un problema
    de parseo (que ya se habría detenido antes). También verifica que ninguna
    orden quede "aprobada" o "entregada" antes de haberse comprado: eso
    invalidaría el cálculo de Recency en RFM.
    """
    missing_delivery = orders["order_delivered_customer_date"].is_null()

    result = _parse_dates(orders, ORDER_DATE_COLUMNS)

    if result.filter(pl.col("order_purchase_timestamp").is_null()).height > 0:
        raise ContractViolation(
            "silver.orders no puede tener order_purchase_timestamp ausente."
        )

    for column in ("order_approved_at", "order_delivered_customer_date"):
        rota = result.filter(
            pl.col(column).is_not_null()
            & (pl.col(column) < pl.col("order_purchase_timestamp"))
        )
        if rota.height > 0:
            raise ContractViolation(
                f"silver.orders rompe la secuencia temporal: {column} es "
                "anterior a order_purchase_timestamp."
            )

    return result.with_columns(
        missing_delivery.alias("delivery_timestamp_missing")
    )


def build_customers(customers: pl.DataFrame) -> pl.DataFrame:
    """Conserva clientes y verifica la relación uno a uno con customer_id."""
    if customers.filter(pl.col("customer_id").is_null()).height > 0:
        raise ContractViolation(
            "silver.customers no puede tener customer_id nulo."
        )

    if customers["customer_id"].n_unique() != customers.height:
        raise ContractViolation(
            "silver.customers no respeta la relación uno a uno de customer_id."
        )

    return customers


def build_order_items(items: pl.DataFrame) -> pl.DataFrame:
    """Comprueba que los ítems no tengan precios ni fletes negativos."""
    negativos = items.filter(
        (pl.col("price") < 0) | (pl.col("freight_value") < 0)
    )
    if negativos.height > 0:
        raise ContractViolation(
            "silver.order_items contiene precios o fletes negativos."
        )

    return items


def build_payments(payments: pl.DataFrame) -> pl.DataFrame:
    """Comprueba que los pagos no tengan montos negativos."""
    valores = payments["payment_value"]

    no_finitos = payments.filter(~valores.is_finite())
    if no_finitos.height > 0:
        raise ContractViolation("silver.payments contiene montos no finitos.")

    negativos = payments.filter(pl.col("payment_value") < 0)
    if negativos.height > 0:
        raise ContractViolation("silver.payments contiene montos negativos.")

    return payments


def validate_relationships(
    orders: pl.DataFrame,
    customers: pl.DataFrame,
    items: pl.DataFrame,
    payments: pl.DataFrame,
) -> None:
    """Verifica las claves foráneas antes de construir productos Gold."""
    order_ids = set(orders["order_id"].to_list())
    customer_ids = set(customers["customer_id"].to_list())

    huerfanas_customer = orders.filter(
        ~pl.col("customer_id").is_in(customer_ids)
    )
    if huerfanas_customer.height > 0:
        raise ContractViolation(
            "silver.orders tiene una clave foránea huérfana hacia "
            "silver.customers."
        )

    huerfanas_items = items.filter(~pl.col("order_id").is_in(order_ids))
    if huerfanas_items.height > 0:
        raise ContractViolation(
            "silver.order_items tiene una clave foránea huérfana hacia "
            "silver.orders."
        )

    huerfanas_payments = payments.filter(~pl.col("order_id").is_in(order_ids))
    if huerfanas_payments.height > 0:
        raise ContractViolation(
            "silver.payments tiene una clave foránea huérfana hacia "
            "silver.orders."
        )
