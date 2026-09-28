"""Agregaciones sobre las temperaturas medias mensuales."""

from __future__ import annotations

import polars as pl


def resumen_mensual(
    mensuales: pl.DataFrame | pl.LazyFrame,
    paises: list[str] | tuple[str, ...] | None = None,
) -> pl.DataFrame | pl.LazyFrame:
    datos = mensuales

    if paises is not None:
        datos = datos.filter(pl.col("iso_alpha3").is_in(list(paises)))

    resultado = (
        datos.group_by(["iso_alpha3", "country", "month"])
        .agg(
            pl.len().alias("observaciones"),
            pl.col("temperature_c").mean().round(2).alias("temperature_mean"),
        )
        .sort(["iso_alpha3", "month"])
    )

    return resultado


def resumen_anual_desde_mensuales(
    mensuales: pl.DataFrame | pl.LazyFrame,
    paises: list[str] | tuple[str, ...] | None = None,
) -> pl.DataFrame | pl.LazyFrame:
    datos = mensuales

    if paises is not None:
        datos = datos.filter(pl.col("iso_alpha3").is_in(list(paises)))

    resultado = (
        datos.group_by(["iso_alpha3", "country", "year"])
        .agg(
            pl.col("month").n_unique().alias("meses_disponibles"),
            pl.col("temperature_c").mean().round(2).alias("temperature_mean"),
        )
        .sort(["iso_alpha3", "year"])
    )

    return resultado


def anomalias_mensuales(
    mensuales: pl.DataFrame | pl.LazyFrame,
    umbral: float = 2.0,
) -> pl.DataFrame | pl.LazyFrame:
    resultado = (
        mensuales.with_columns(
            pl.col("temperature_c")
            .mean()
            .over(["iso_alpha3", "month"])
            .alias("temperature_mean_month")
        )
        .with_columns(
            pl.when(
                pl.col("temperature_c").std().over(["iso_alpha3", "month"]) > 0
            )
            .then(
                (pl.col("temperature_c") - pl.col("temperature_mean_month"))
                / pl.col("temperature_c").std().over(["iso_alpha3", "month"])
            )
            .otherwise(0.0)
            .fill_null(0.0)
            .alias("standardized_anomaly")
        )
        .with_columns(
            (pl.col("standardized_anomaly").abs() > umbral).alias("is_anomaly")
        )
    )

    return resultado
