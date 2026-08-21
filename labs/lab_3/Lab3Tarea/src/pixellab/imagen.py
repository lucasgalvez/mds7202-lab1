"""Clase ``Imagen``: contenedor de imágenes sobre el que se opera con NumPy."""

from __future__ import annotations

import numpy as np


class Imagen:
    """Contenedor de imágenes RGB.

    Completen el constructor y los operadores de esta clase siguiendo el
    contrato del enunciado y los tests de ``tests/test_imagen.py``.
    """

    def __init__(self, img: np.ndarray):

        if not isinstance(img, np.ndarray):
            raise TypeError(
                "Debes entregar un arreglo de numpy como argumento del constructor de Imagen"
            )
        if img.ndim != 3:
            raise ValueError("La imagen debe tener 3 dimensiones")
        if img.shape[-1] != 3:
            raise ValueError("La imagen debe tener 3 canales")

        self.imagen = img

    def __add__(self, other: int | float | np.ndarray | Imagen) -> Imagen:
        if isinstance(other, Imagen):  # revisa si es una imagen el other
            if self.imagen.shape != other.imagen.shape:
                raise ValueError("Las dimensiones de las imágenes no calzan")
            otro = other.imagen
        else:
            otro = other

        resultado = self.imagen + otro
        resultado = np.clip(
            resultado, 0, 255
        )  # limita los valores entre 0 y 255
        resultado = resultado.astype(int)
        return Imagen(
            np.copy(resultado)
        )  # hace una copia de la imagen resultante

    def __radd__(self, other: int | float | np.ndarray | Imagen) -> Imagen:
        return self.__add__(
            other
        )  # usa el metodo __add__ para la suma reflejada

    def __sub__(self, other: int | float | np.ndarray | Imagen) -> Imagen:
        if isinstance(other, Imagen):
            if self.imagen.shape != other.imagen.shape:
                raise ValueError("Las dimensiones de las imágenes no calzan")
            otro = other.imagen
        else:
            otro = other

        resultado = self.imagen - otro
        resultado = np.clip(resultado, 0, 255)
        resultado = resultado.astype(int)
        return Imagen(np.copy(resultado))

    def __rsub__(self, other: int | float | np.ndarray | Imagen) -> Imagen:
        if isinstance(other, Imagen):
            if self.imagen.shape != other.imagen.shape:
                raise ValueError("Las dimensiones de las imágenes no calzan")
            otro = other.imagen
        else:
            otro = other

        resultado = otro - self.imagen
        resultado = np.clip(resultado, 0, 255)
        resultado = resultado.astype(int)
        return Imagen(np.copy(resultado))

    def __mul__(self, other: int | float | np.ndarray | Imagen) -> Imagen:
        if isinstance(other, Imagen):
            if self.imagen.shape != other.imagen.shape:
                raise ValueError("Las dimensiones de las imágenes no calzan")
            otro = other.imagen
        else:
            otro = other

        resultado = self.imagen * otro
        resultado = np.clip(resultado, 0, 255)
        resultado = resultado.astype(int)
        return Imagen(np.copy(resultado))

    def __rmul__(self, other: int | float | np.ndarray | Imagen) -> Imagen:
        return self.__mul__(other)
