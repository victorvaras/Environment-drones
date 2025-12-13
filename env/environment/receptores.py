from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class Receptor:
    """Clase de datos para posición inicial de los receptores."""
    x: float
    y: float
    z: float

class ReceptoresManager:
    def __init__(self, receptores: list[Receptor]):
        self._rx = receptores

    @property
    def n(self) -> int:
        return len(self._rx)

    def positions_xyz(self) -> np.ndarray:
        return np.array([[r.x, r.y, r.z] for r in self._rx], dtype=float)