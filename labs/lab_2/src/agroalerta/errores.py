class LecturaInvalidaError(Exception):
    """El valor medido es físicamente imposible."""


class DatosInsuficientesError(Exception):
    """No hay suficientes lecturas para concluir algo."""
