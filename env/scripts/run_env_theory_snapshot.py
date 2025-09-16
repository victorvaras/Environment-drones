# === Bootstrap sys.path a la raíz del proyecto (dos niveles arriba) ===
import sys, os, time
from pathlib import Path
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Forzar backend interactivo para que se abra la ventana
os.environ.pop("MPLBACKEND", None)
import matplotlib
matplotlib.use("TkAgg", force=True)

import numpy as np
from env.environment.gymnasium_env import DroneEnv

# === Configuración del escenario (edita aquí si quieres) ===
SCENE = "simple_street_canyon_with_cars"
DRONE_START = (0.0, 0.0, 20.0)
RX_POSITIONS = [
    (-50.0, 0.0, 1.5),
    (0.0,   30.0, 1.5),
    ( 20.0,  -30.0, 1.5),
    (80.0,   40.0, 1.5),
    (  50.0,    0.0, 1.5),
    (90, -55, 1.5),
]
ANTENNA_MODE = "ISO"

# RF
FREQ_MHZ     = 3_500 #28_000.0     # Frecuencia en MHz de transmisor
TX_POWER_DBM = 30.0               # Potencia de transmisión en dBm
NOISE_FIG_DB = 7.0                # Figura de ruido en dB
B_HZ         = 20e6               # Ancho de banda en Hz

# Parámetros Goldsmith cálculo teórico
GAMMA = 2.0                     # Exponente de pérdida (2 en espacio libre) (2.7-3.5 en ciudad pequeña) (valores en libros)
D0_M  = 1.0                     # Distancia de referencia en metros, para calibracion 
GT_DBI = 0.0                    # Ganancia antena TX en dBi
GR_DBI = 0.0                    # Ganancia antena RX en dBi

if __name__ == "__main__":
    start_time = time.perf_counter()

    env = DroneEnv(
        render_mode="human",            # no usaremos el render interno, pero deja 'human'
        scene_name=SCENE,
        max_steps=1,                    # sin movimiento
        drone_start=DRONE_START,
        rx_positions=RX_POSITIONS,
        antenna_mode=ANTENNA_MODE,      # "ISO" o "SECTOR3_3GPP"
        frequency_mhz=FREQ_MHZ,
        tx_power_dbm=TX_POWER_DBM,
        noise_figure_db=NOISE_FIG_DB,
        bandwidth_hz=B_HZ,
    )

    # Inicializa escena (no step)
    obs, info = env.reset(seed=0)

    # PRx teórico (Goldsmith 2.40) y PRx real (Sionna RT)
    prx_teo  = env.rt.compute_prx_dbm_theoretical(gamma=GAMMA, d0=D0_M, Gt_dBi=GT_DBI, Gr_dBi=GR_DBI)
    prx_real = env.rt.compute_prx_dbm()   # tu método existente basado en RT

    # Render estático dual
    env.render_dual_snapshot(prx_teo, prx_real)

    env.close()
    elapsed = time.perf_counter() - start_time
    print(f"Snapshot dual listo. Tiempo total: {elapsed:.3f} s")
