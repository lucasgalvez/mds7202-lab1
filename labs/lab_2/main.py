"""Orquestador del mini proyecto AgroAlerta."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
from agroalerta.datos import cargar_lecturas
from agroalerta.errores import DatosInsuficientesError
from agroalerta.reporte import contar_riesgos
from agroalerta.sensores import SensorHumedad, SensorTemperatura, SensorViento


def crear_sensores() -> list:
    """Crea los sensores con los umbrales de configuración de AgroAlerta."""
    return [
        SensorTemperatura(0, 40),
        SensorViento(25),
        SensorHumedad(85),
    ]


def imprimir_reporte(fecha: str, conteo: dict, descartadas: int) -> None:
    """Imprime el reporte de riesgos por sensor para una fecha dada."""
    print(f"Estación Parcela Norte — {fecha}")
    total = 0
    for nombre, cantidad in conteo.items():
        print(f"{nombre.capitalize()} {cantidad} lecturas en riesgo")
        total += cantidad
    print(f"Descartadas: {descartadas} lecturas inválidas")
    print(f"Total: {total} situaciones de riesgo")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AgroAlerta - reporte de riesgos"
    )
    parser.add_argument(
        "--fecha", required=True, help="Fecha en formato YYYY-MM-DD"
    )
    args = parser.parse_args()

    sensores = crear_sensores()
    ruta_csv = Path(__file__).parent / "data" / "lecturas.csv"
    lecturas = cargar_lecturas(ruta_csv, args.fecha)

    try:
        conteo, descartadas = contar_riesgos(sensores, lecturas)
    except DatosInsuficientesError as error:
        print(f"No se puede generar el reporte: {error}")
        return

    imprimir_reporte(args.fecha, conteo, descartadas)


if __name__ == "__main__":
    main()
