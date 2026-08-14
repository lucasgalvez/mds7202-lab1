import pytest
from agroalerta.errores import DatosInsuficientesError, LecturaInvalidaError
from agroalerta.reporte import contar_riesgos
from agroalerta.sensores import SensorTemperatura, SensorViento


def test_temperatura_bajo_cero_es_riesgosa():
    sensor = SensorTemperatura(0, 40)
    assert sensor.es_riesgo(-2) is True


def test_viento_normal_no_es_riesgoso():
    sensor = SensorViento(25)
    assert sensor.es_riesgo(10) is False


def test_lectura_fisicamente_imposible_levanta_error():
    sensor = SensorTemperatura(0, 40)
    with pytest.raises(LecturaInvalidaError):
        sensor.validar(-300)


def test_menos_de_20_lecturas_validas_levanta_error():
    sensor = SensorTemperatura(0, 40)
    lecturas = {"temperatura": [18] * 5}  # solo 5 lecturas
    with pytest.raises(DatosInsuficientesError):
        contar_riesgos([sensor], lecturas)
