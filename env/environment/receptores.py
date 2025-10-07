from __future__ import annotations
from dataclasses import dataclass
import numpy as np





@dataclass
class Receptor:
    x: float
    y: float
    z: float

    def step_delta(self, delta_xyz: np.ndarray, bounds = ((-200.0, 200.0), (-200.0, 200.0), (0.0, 120.0)),
                   max_delta = 5.0):
        delta = np.clip(np.array(delta_xyz, dtype = float), -max_delta, max_delta)
        self.x = float(np.clip(self.x + delta[0], bounds[0][0], bounds[0][1]))
        self.y = float(np.clip(self.y + delta[1], bounds[1][0], bounds[1][1]))
        self.z = float(np.clip(self.z + delta[2], bounds[2][0], bounds[2][1]))
        return np.array([self.x, self.y, self.z], dtype = float)

class ReceptoresManager:
    #Metodo que recibe una lista de receptores y los guarda en rx
    def __init__(self, receptores: list[Receptor]):
        self._rx = receptores

    #Metodo que mueve todos los receptores de la misma manera
    def step_all(self, delta_xyz: np.ndarray):
        return np.array([r.step_delta(delta_xyz) for r in self._rx], dtype = float)
    
    @property
    def n(self) -> int:
        return len(self._rx)

    def positions_xyz(self) -> np.ndarray:
        return np.array([[r.x, r.y, r.z] for r in self._rx], dtype=float)
