"""Operaciones de procesamiento de imágenes para completar."""

from __future__ import annotations

import numpy as np
from scipy.signal import convolve2d

from src.pixellab.imagen import Imagen


class LibImagen:
    """Filtros y transformaciones que reciben y retornan ``Imagen``."""

    def to_negative(self, img_in: Imagen) -> Imagen:
        resultado = (
            255 - img_in.imagen
        )  # crea una nueva imagen nueva con los valores negativos
        resultado = resultado.astype(int)

        return Imagen(resultado)

    def to_gray(self, img_in: Imagen) -> Imagen:
        r = img_in.imagen[:, :, 0]
        g = img_in.imagen[:, :, 1]
        b = img_in.imagen[:, :, 2]

        gris = 0.299 * r + 0.587 * g + 0.114 * b

        resultado = np.stack(
            [gris, gris, gris], axis=2
        )  # hace que el resultado tenga 3 canales
        resultado = resultado.astype(int)

        return Imagen(resultado)

    def get_channel(self, img_in: Imagen, channel: str) -> Imagen:
        canales = {"r": 0, "g": 1, "b": 2}
        if channel not in canales:  # revisa que el canal sea r, g o b
            raise ValueError(
                f"Canal '{channel}' no válido. Valores posibles: 'r', 'g' o 'b'."
            )
        resultado = np.zeros_like(img_in.imagen)
        indice = canales[channel]
        resultado[:, :, indice] = img_in.imagen[:, :, indice]
        resultado = resultado.astype(int)

        return Imagen(resultado)

    def flip(self, img_in: Imagen, axis: str) -> Imagen:
        if axis == "h":
            resultado = img_in.imagen[:, ::-1, :]
        elif axis == "v":
            resultado = img_in.imagen[::-1, :, :]
        else:
            raise ValueError(
                f"Eje '{axis}' no válido. Valores posibles: 'h' (horizontal) o 'v' (vertical)."
            )

        resultado = np.copy(resultado).astype(int)
        return Imagen(resultado)

    def set_saturation(self, img_in: Imagen, C: float) -> Imagen:
        img = img_in.imagen.astype(
            float
        )  # transforma img a float para poder operar
        gris = self.to_gray(img_in).imagen.astype(
            float
        )  # transforma gris a float para poder operar

        resultado = gris + C * (img - gris)  # aplica la formula de saturacion

        resultado[resultado < 0] = 0  # limita los valores menores a 0 a 0
        resultado[resultado > 255] = (
            255  # limita los valores mayores a 255 a 255
        )

        resultado = resultado.astype(
            int
        )  # vuelve a transformar resultado a int para que sea una imagen valida

        return Imagen(resultado)

    def set_contrast(self, img_in: Imagen, C: float) -> Imagen:
        img = img_in.imagen.astype(float)

        F = 259 * (C + 255) / (255 * (259 - C))
        resultado = F * (img - 128) + 128

        resultado[resultado < 0] = 0
        resultado[resultado > 255] = 255

        resultado = resultado.astype(int)

        return Imagen(resultado)

    def conv_channel(self, img_in: Imagen, kernel: np.ndarray) -> Imagen:
        """Por documentar (esto es parte del trabajo de la Etapa 6)."""
        # El cuerpo de este método lo entrega el curso.
        img = img_in.imagen
        img_out = []
        for i in range(img.shape[-1]):
            img_channel = convolve2d(
                img[:, :, i], kernel, mode="same", boundary="symm"
            )
            img_out.append(img_channel)
        new_image = np.stack(img_out, axis=2)
        new_image[new_image > 255], new_image[new_image < 0] = 255, 0
        return Imagen(new_image.astype(int))
