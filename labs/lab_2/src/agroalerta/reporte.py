from agroalerta.errores import DatosInsuficientesError, LecturaInvalidaError


def contar_riesgos(sensores, lecturas):
    conteo = {}
    descartadas = 0

    for sensor in sensores:
        valores = lecturas.get(sensor.nombre, [])
        riesgos = 0
        validas = 0

        for valor in valores:
            try:
                sensor.validar(valor)
            except LecturaInvalidaError:
                descartadas += 1
                continue

            validas += 1
            if sensor.es_riesgo(valor):
                riesgos += 1

        if validas < 20:
            raise DatosInsuficientesError(
                f"Menos de 20 lecturas válidas para {sensor.nombre}"
            )

        conteo[sensor.nombre] = riesgos

    return conteo, descartadas
