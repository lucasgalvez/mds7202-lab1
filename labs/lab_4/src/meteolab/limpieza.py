"""Funciones para revisar nulos y claves temporales."""

from __future__ import annotations

import polars as pl

from src.meteolab.constantes import Tabla


def resumen_de_nulos(temperaturas: pl.DataFrame) -> pl.DataFrame:
    total_filas = temperaturas.height
    resumen = []

    for columna in temperaturas.columns:
        cantidad_nulos = temperaturas[columna].null_count()
        porcentaje_nulos = cantidad_nulos / total_filas * 100

        resumen.append(
            {
                "columna": columna,
                "nulos": cantidad_nulos,
                "porcentaje": porcentaje_nulos,
            }
        )

    return pl.DataFrame(resumen)


def claves_repetidas(temperaturas: Tabla) -> Tabla:
    repetidas = (
        temperaturas.group_by(["country", "year", "period"])
        .agg(pl.len())
        .filter(pl.col("len") > 1)
    )

    return repetidas


def limpiar_temperaturas(temperaturas: Tabla) -> Tabla:
    periodos_mensuales = [
        "JAN",
        "FEB",
        "MAR",
        "APR",
        "MAY",
        "JUN",
        "JUL",
        "AUG",
        "SEP",
        "OCT",
        "NOV",
        "DEC",
    ]

    limpias = temperaturas.filter(
        pl.col("period").is_in(periodos_mensuales)
        & pl.col("temperature_c").is_not_null()
    )

    return limpias
