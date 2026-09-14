"""Ingesta reproducible de las fuentes Parquet hacia Bronze."""

from __future__ import annotations

from pathlib import Path

import polars as pl


def read_sources(raw_dir: Path) -> dict[str, pl.DataFrame]:
    """Lee las cuatro fuentes crudas y conserva exactamente su esquema."""

    sources = {
        "orders": "orders.parquet",
        "customers": "customers.parquet",
        "order_items": "order_items.parquet",
        "payments": "payments.parquet",
    }

    result: dict[str, pl.DataFrame] = {}

    for name, filename in sources.items():
        path = raw_dir / filename

        if not path.exists():
            raise FileNotFoundError(
                f"No se encontró la fuente '{name}': {path}"
            )

        result[name] = pl.read_parquet(path)

    return result
