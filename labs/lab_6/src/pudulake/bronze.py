"""Ingesta reproducible de las fuentes Parquet hacia Bronze."""

from __future__ import annotations

from pathlib import Path

import polars as pl

SOURCES = ("orders", "customers", "order_items", "payments")


def read_sources(raw_dir: Path) -> dict[str, pl.DataFrame]:
    """Lee las cuatro fuentes crudas y conserva exactamente su esquema."""
    sources: dict[str, pl.DataFrame] = {}

    for name in SOURCES:
        path = raw_dir / f"{name}.parquet"

        if not path.exists():
            raise FileNotFoundError(f"Falta la fuente Bronze '{name}': {path}")

        sources[name] = pl.read_parquet(path)

    return sources
