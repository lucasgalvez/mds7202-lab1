"""Pipelines lazy para analizar únicamente temperaturas mensuales."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from src.meteolab.constantes import PAISES_COMPARACION, RUTA_CSV


def pipeline_mensual(
    ruta: Path = RUTA_CSV,
    paises: list[str] | tuple[str, ...] = PAISES_COMPARACION,
) -> pl.LazyFrame:
    from src.meteolab.constantes import ESQUEMA_CRU
    from src.meteolab.derivadas import agregar_fecha_mensual
    from src.meteolab.limpieza import limpiar_temperaturas

    datos = pl.scan_csv(
        ruta,
        schema_overrides=ESQUEMA_CRU,
    )

    datos = limpiar_temperaturas(datos)

    datos = datos.filter(pl.col("iso_alpha3").is_in(list(paises)))

    datos = agregar_fecha_mensual(datos)

    datos = datos.sort(["country", "fecha"])

    return datos


def pipeline_resumen_mensual(
    ruta: Path = RUTA_CSV,
    paises: list[str] | tuple[str, ...] = PAISES_COMPARACION,
) -> pl.LazyFrame:
    from src.meteolab.metricas import resumen_mensual

    mensuales = pipeline_mensual(ruta, paises)

    resultado = resumen_mensual(mensuales)

    return resultado


def pipeline_resumen_anual(
    ruta: Path = RUTA_CSV,
    paises: list[str] | tuple[str, ...] = PAISES_COMPARACION,
) -> pl.LazyFrame:
    from src.meteolab.metricas import resumen_anual_desde_mensuales

    mensuales = pipeline_mensual(ruta, paises)

    resultado = resumen_anual_desde_mensuales(mensuales)

    return resultado


def pipeline_anomalias(
    ruta: Path = RUTA_CSV,
    paises: list[str] | tuple[str, ...] = PAISES_COMPARACION,
    umbral: float = 2.0,
) -> pl.LazyFrame:
    from src.meteolab.metricas import anomalias_mensuales

    mensuales = pipeline_mensual(ruta, paises)

    resultado = anomalias_mensuales(
        mensuales,
        umbral=umbral,
    )

    return resultado


def ejecutar_reporte(
    ruta: Path = RUTA_CSV,
    paises: list[str] | tuple[str, ...] = PAISES_COMPARACION,
) -> pl.DataFrame:
    consulta = pipeline_resumen_mensual(ruta, paises)

    resultado = consulta.collect()

    return resultado


def plan_de_ejecucion(
    ruta: Path = RUTA_CSV,
    paises: list[str] | tuple[str, ...] = PAISES_COMPARACION,
    optimizado: bool = True,
) -> str:
    consulta = pipeline_mensual(ruta, paises)

    return consulta.explain(optimized=optimizado)
