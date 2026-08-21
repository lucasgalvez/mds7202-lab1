"""Kernels de convolución que deben definir para la Etapa 6."""

import numpy as np

# Su código aquí: agreguen al menos cinco tuplas (nombre, kernel).

KERNELS: list[tuple[str, np.ndarray]] = [
    (
        "identidad",
        np.array(
            [
                [0, 0, 0],
                [0, 1, 0],
                [0, 0, 0],
            ]
        ),
    ),
    (
        "laplaciano",
        np.array(
            [
                [0, -1, 0],
                [-1, 4, -1],
                [0, -1, 0],
            ]
        ),
    ),
    (
        "enfoque",
        np.array(
            [
                [0, -1, 0],
                [-1, 5, -1],
                [0, -1, 0],
            ]
        ),
    ),
    (
        "desenfoque",
        np.array(
            [
                [1 / 9, 1 / 9, 1 / 9],
                [1 / 9, 1 / 9, 1 / 9],
                [1 / 9, 1 / 9, 1 / 9],
            ]
        ),
    ),
    (
        "relieve",
        np.array(
            [
                [-2, -1, 0],
                [-1, 1, 1],
                [0, 1, 2],
            ]
        ),
    ),
]
