"""Lectura y transformación de la tabla de mediciones."""

from __future__ import annotations

import polars as pl

try:
    from .contratos import OBJETIVOS
except ImportError:  # pragma: no cover
    OBJETIVOS = [
        "Dry_Green_g",
        "Dry_Dead_g",
        "Dry_Clover_g",
        "GDM_g",
        "Dry_Total_g",
    ]

_COLUMNAS_LARGO = ("sample_id", "target_name", "target")


def a_formato_ancho(df_largo: pl.DataFrame) -> pl.DataFrame:
    """Pasa la tabla de formato largo a ancho: una fila por imagen.

    El índice son todas las columnas salvo ``sample_id``, ``target_name`` y
    ``target``; ``sample_id`` es distinto en cada una de las cinco filas de una
    imagen y, si entrara al índice, esas filas no se juntarían.
    """
    indice = [c for c in df_largo.columns if c not in _COLUMNAS_LARGO]
    ancho = df_largo.pivot(
        on="target_name",
        index=indice,
        values="target",
        aggregate_function="first",
    )
    return ancho.select(*indice, *OBJETIVOS)
