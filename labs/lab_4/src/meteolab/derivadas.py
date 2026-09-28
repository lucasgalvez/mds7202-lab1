"""Funciones para construir fechas mensuales."""

from __future__ import annotations

import polars as pl

from src.meteolab.constantes import Tabla


def agregar_fecha_mensual(mensuales: Tabla) -> Tabla:
    from src.meteolab.constantes import MESES

    resultado = mensuales.with_columns(
        pl.col("period").replace(MESES).cast(pl.Int8).alias("month")
    ).with_columns(
        pl.date(
            pl.col("year"),
            pl.col("month"),
            1,
        ).alias("fecha")
    )

    return resultado
