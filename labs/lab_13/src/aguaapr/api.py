"""Rutas de inferencia: información del modelo, vida y predicción."""

from fastapi import APIRouter, Depends

from src.aguaapr.artifacts import a_tabla, predecir
from src.aguaapr.constants import FEATURES, OPCIONALES, RANGOS
from src.aguaapr.contracts import RespuestaPrediccion, Servicios
from src.aguaapr.deps import crear_dependencia_clave
from src.aguaapr.validation import MedicionAgua


def crear_router_inferencia(servicios: Servicios) -> APIRouter:
    """Construye el router con tres rutas.

    - `GET /`: describe el modelo. Responde las métricas de validación de la
      metadata, la entrada (cada columna con su rango) y la
      salida.
    - `GET /health`: ruta de vida, pública. Responde
      `{"estado": "ok"}`.
    - `POST /predict`: recibe una `MedicionAgua` en el cuerpo y responde una
      `RespuestaPrediccion` con la predicción y su probabilidad. Exige la
      clave de API con `crear_dependencia_clave`.
    """
    router = APIRouter()
    exigir_clave = crear_dependencia_clave(servicios)

    @router.get("/")
    def informacion() -> dict:
        return {
            "metricas_validacion": servicios.metadata.get(
                "metricas_validacion"
            ),
            "entrada": {
                c: {
                    "minimo": RANGOS[c][0],
                    "maximo": RANGOS[c][1],
                    "opcional": c in OPCIONALES,
                }
                for c in FEATURES
            },
            "salida": {
                "prediccion": "1 si el agua es potable, 0 si no lo es",
                "probabilidad": "probabilidad de que el agua sea potable",
            },
        }

    @router.get("/health")
    def health() -> dict:
        return {"estado": "ok"}

    @router.post("/predict", dependencies=[Depends(exigir_clave)])
    def predict(medicion: MedicionAgua) -> RespuestaPrediccion:
        clases, probabilidades = predecir(
            servicios.modelo, a_tabla([medicion])
        )
        return RespuestaPrediccion(
            prediccion=int(clases[0]),
            probabilidad=round(float(probabilidades[0]), 4),
        )

    return router
