"""Funciones para declarar y validar el esquema CRU."""

from __future__ import annotations

import pandera.polars as pa
import polars as pl

from src.meteolab.constantes import PERIODOS_VALIDOS

ESQUEMA_TEMPERATURAS = pa.DataFrameSchema(
    {
        "country": pa.Column(pl.String),
        "iso_alpha2": pa.Column(pl.String),
        "iso_alpha3": pa.Column(pl.String),
        "year": pa.Column(
            pl.Int64,
            checks=pa.Check.in_range(1901, 2025),
        ),
        "period": pa.Column(
            pl.String,
            checks=pa.Check.isin(PERIODOS_VALIDOS),
        ),
        "temperature_c": pa.Column(
            pl.Float64,
            nullable=True,
        ),
        "parameter": pa.Column(
            pl.String,
            checks=pa.Check.eq("Mean Temperature"),
        ),
        "units": pa.Column(
            pl.String,
            checks=pa.Check.eq("degrees Celsius"),
        ),
        "source_file": pa.Column(pl.String),
    }
)


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
