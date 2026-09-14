"""Funciones para declarar y validar el esquema CRU."""

from __future__ import annotations

import pandera.polars as pa
import polars as pl

ESQUEMA_TEMPERATURAS = pa.DataFrameSchema({})

ESQUEMA_ESPERADO = {
    "country": pl.String,
    "iso_alpha2": pl.String,
    "iso_alpha3": pl.String,
    "year": pl.Int64,
    "period": pl.String,
    "temperature_c": pl.Float64,
    "parameter": pl.String,
    "units": pl.String,
    "source_file": pl.String,
}


def comparar_esquema(temperaturas: pl.DataFrame) -> list[str]:
    diferencias = []

    for columna, tipo_esperado in ESQUEMA_ESPERADO.items():
        if columna not in temperaturas.columns:
            diferencias.append(f"Falta la columna {columna}")

        elif temperaturas.schema[columna] != tipo_esperado:
            diferencias.append(
                f"{columna}: se esperaba {tipo_esperado}, "
                f"pero se encontró {temperaturas.schema[columna]}"
            )

    return diferencias


def validar_esquema(temperaturas: pl.DataFrame) -> None:
    diferencias = comparar_esquema(temperaturas)

    if diferencias:
        raise ValueError("; ".join(diferencias))


def validar_datos(temperaturas: pl.DataFrame) -> pl.DataFrame:
    validar_esquema(temperaturas)
    validado = ESQUEMA_TEMPERATURAS.validate(temperaturas)
    return validado


def casos_que_fallan(temperaturas: pl.DataFrame) -> pl.DataFrame:
    try:
        ESQUEMA_TEMPERATURAS.validate(temperaturas, lazy=True)

    except pa.errors.SchemaErrors as error:
        return error.failure_cases

    return pl.DataFrame()
