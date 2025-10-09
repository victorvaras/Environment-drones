# === Bootstrap sys.path a la raíz del proyecto (dos niveles arriba) ===
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import os
import numpy as np
from gymnasium.wrappers import RecordVideo
from env.environment.gymnasium_env import DroneEnv  # <- usa tu archivo renombrado

import time

# === Configuración del escenario (edita aquí) ===
SCENE = "simple_street_canyon_with_cars"                  # nombre integrado o ruta a XML/carpeta
DRONE_START = (0.0, 0.0, 20.0)    # (x, y, z) en metros
RX_POSITIONS = [
    #(-50.0, 0.0, 1.5),
    #(0.0,   30.0, 1.5),
    #( 20.0,  -30.0, 1.5),
    #(80.0,   40.0, 1.5),
    #(  50.0,    0.0, 1.5),
    #(90, -55, 1.5),

    (20.0, -30.0, 1.5),
    (10.0, 0.0, 1.5),
    (-10.0, 0.0, 1.5),
]


MAX_STEPS = 100

# Carpeta donde se guardarán los videos
OUTDIR = Path(__file__).resolve().parent / "videos"


if __name__ == "__main__":

    start_time = time.perf_counter()

    os.makedirs(OUTDIR, exist_ok=True)

    env = DroneEnv(
        render_mode="rgb_array",   # requerido por RecordVideo
        scene_name=SCENE,
        max_steps=MAX_STEPS,
        drone_start=DRONE_START,
        rx_positions=RX_POSITIONS if RX_POSITIONS else None,
        antenna_mode="SECTOR3_3GPP",  # "ISO" o "SECTOR3_3GPP"
    )
    env = RecordVideo(env, str(OUTDIR), name_prefix="metricas_avanzadas-1")

    obs, info = env.reset(seed=0)
    done, trunc = False, False
    while not (done or trunc):
        #a = np.random.uniform(-2, 2, size=(3,))    #Movimiento aleatorio
        a = [0,0,0]     #Movimiento estatico
        b = [0,0,0]     #Sin movimiento receptores
        obs, rew, done, trunc, info = env.step(a)


    elapsed = time.perf_counter() - start_time
    print(f"Tiempo total transcurrido: {elapsed:.3f} s ({elapsed/60:.2f} min)")


    env.close()
    print(f"Video guardado en: {OUTDIR}")
