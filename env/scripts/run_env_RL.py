# === Bootstrap sys.path a la raíz del proyecto (dos niveles arriba) ===
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Forzar backend interactivo para que se abra la ventana
import os
os.environ.pop("MPLBACKEND", None)     # por si estaba forzado a 'Agg'
import matplotlib
matplotlib.use("TkAgg", force=True)    # o 'Qt5Agg' si prefieres

import numpy as np
from time import perf_counter
from env.environment.gymnasium_env import DroneEnv  # <- usa tu archivo renombrado



# === Configuración del escenario (edita aquí) ===
SCENE = "simple_street_canyon_with_cars"  # municg - san_francisco - simple_street_canyon - simple_street_canyon_with_cars
DRONE_START = (0.0, 0.0, 20.0)    # (x, y, z) en metros
RX_POSITIONS = [
    (-50.0, 0.0, 1.5),
    (0.0,   30.0, 1.5),
    ( 20.0,  -30.0, 1.5),
    (80.0,   40.0, 1.5),
    (  50.0,    0.0, 1.5),
    (90, -55, 1.5),
]
MAX_STEPS = 10


if __name__ == "__main__":

    env = DroneEnv(
        scene_name=SCENE,
        max_steps=MAX_STEPS,
        drone_start=DRONE_START,
        rx_positions=RX_POSITIONS if RX_POSITIONS else None,
        antenna_mode="SECTOR3_3GPP",  # "ISO" o "SECTOR3_3GPP"
    )

    start_time = perf_counter()
    obs, info = env.reset(seed=0)
    done, trunc = False, False

    while not (done or trunc):
        a = [4, 0, 0]
        b = [0, 0, 0]
        obs, rew, done, trunc, info = env.step(a,b)

    end_time = perf_counter()
    elapsed = end_time - start_time
    print(f"Tiempo total episodio, 100 steps: 500mil rayos {elapsed:.3f} s")

    env.close()
