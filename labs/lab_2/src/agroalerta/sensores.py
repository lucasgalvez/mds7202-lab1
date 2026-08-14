from abc import ABC, abstractmethod

from agroalerta.errores import LecturaInvalidaError


class Sensor(ABC):
    """Clase base abstracta para los sensores de la estación."""

    def __init__(self, nombre: str, unidad: str) -> None:
        self.nombre = nombre
        self.unidad = unidad

    @abstractmethod
    def es_riesgo(self, valor: float) -> bool: ...

    @property
    @abstractmethod
    def rango_fisico(self) -> tuple[float, float]: ...

    @property
    def rango_seguro(self) -> str:
        raise NotImplementedError

    def validar(self, valor: float) -> None:
        minimo, maximo = self.rango_fisico
        if valor < minimo or valor > maximo:
            raise LecturaInvalidaError(
                f"{valor} {self.unidad} está fuera del rango físico de {self.nombre}"
            )


class SensorTemperatura(Sensor):
    def __init__(self, minimo: float, maximo: float) -> None:
        super().__init__("temperatura", "°C")
        self._minimo = minimo
        self._maximo = maximo

    def es_riesgo(self, valor: float) -> bool:
        return valor < self._minimo or valor > self._maximo

    @property
    def rango_seguro(self) -> str:
        return f"entre {self._minimo} y {self._maximo} {self.unidad}"

    @property
    def rango_fisico(self) -> tuple[float, float]:
        return (-50, 60)


class SensorViento(Sensor):
    def __init__(self, maximo: float) -> None:
        super().__init__("viento", "km/h")
        self._maximo = maximo

    def es_riesgo(self, valor: float) -> bool:
        return valor > self._maximo

    @property
    def rango_seguro(self) -> str:
        return f"bajo {self._maximo} {self.unidad}"

    @property
    def rango_fisico(self) -> tuple[float, float]:
        return (0, 200)


class SensorHumedad(Sensor):
    def __init__(self, maximo: float) -> None:
        super().__init__("humedad", "%")
        self._maximo = maximo

    def es_riesgo(self, valor: float) -> bool:
        return valor > self._maximo

    @property
    def rango_seguro(self) -> str:
        return f"bajo {self._maximo} {self.unidad}"

    @property
    def rango_fisico(self) -> tuple[float, float]:
        return (0, 100)
