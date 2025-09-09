from __future__ import annotations
import numpy as np

class Dron:
    def __init__(
        self,
        start_xyz=(0.0, 0.0, 100.0),
        bounds=((-200.0, 200.0), (-200.0, 200.0), (5.0, 120.0)),
        max_delta=5.0,
    ):
        self.pos = np.array(start_xyz, dtype=float)
        self.bounds = bounds
        self.max_delta = max_delta

    def step_delta(self, delta_xyz: np.ndarray):
        delta = np.clip(np.array(delta_xyz, dtype=float), -self.max_delta, self.max_delta)
        self.pos = self.pos + delta
        # Limita a caja de trabajo
        self.pos[0] = np.clip(self.pos[0], self.bounds[0][0], self.bounds[0][1])
        self.pos[1] = np.clip(self.pos[1], self.bounds[1][0], self.bounds[1][1])
        self.pos[2] = np.clip(self.pos[2], self.bounds[2][0], self.bounds[2][1])
        return self.pos.copy()
