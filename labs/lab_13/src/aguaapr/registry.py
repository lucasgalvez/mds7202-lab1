"""Carga del modelo desde su artefacto."""

from pathlib import Path
from typing import Any

import joblib
import sklearn
from sklearn.pipeline import Pipeline

from src.aguaapr.artifacts import ErrorArtefacto, leer_metadata
from src.aguaapr.constants import FEATURES


def cargar_modelo(directorio: Path) -> tuple[Pipeline, dict[str, Any]]:
    """Carga el artefacto de `directorio` y devuelve (modelo, metadata).

    Antes de cargar el modelo comprueba que las variables de la metadata sean
    exactamente `FEATURES`, en ese orden, y que `sklearn_version` coincida con
    la versión de scikit-learn instalada. Si algo no calza lanza
    `ErrorArtefacto` con un mensaje que nombra la diferencia.
    """
    metadata = leer_metadata(directorio)
    if metadata.get("features") != FEATURES:
        raise ErrorArtefacto(
            f"Las variables del artefacto {metadata.get('features')} "
            f"no coinciden con las del contrato {FEATURES}."
        )
    if metadata.get("sklearn_version") != sklearn.__version__:
        raise ErrorArtefacto(
            f"El artefacto se entrenó con scikit-learn "
            f"{metadata.get('sklearn_version')} y está instalada la "
            f"{sklearn.__version__}."
        )
    modelo = joblib.load(directorio / "model.joblib")
    return modelo, metadata
