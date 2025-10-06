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
import time
from env.environment.gymnasium_env import DroneEnv  # <- usa tu archivo renombrado

import pandas as pd


# === Configuración del escenario (edita aquí) ===

SCENE = "simple_street_canyon_with_cars"  # santiago.xml munich - san_francisco - simple_street_canyon - simple_street_canyon_with_cars


DRONE_START = (0.0, 0.0, 10.0)    # (x, y, z) en metros
RX_POSITIONS = [
    (-50.0, 0.0, 1.5),
    #(0.0,   30.0, 1.5),
    (20.0,  -30.0, 1.5),
    #(80.0,   40.0, 1.5),
    #(50.0,    0.0, 1.5),
    #(90, -55, 1.5),
    (10.0, 0.0, 1.5),
    (-10.0, 0.0, 1.5),
    (-1.0, 0.0, 1.5),
]
MAX_STEPS = 50


if __name__ == "__main__":
    

    env = DroneEnv(
        render_mode="human",
        scene_name=SCENE,
        max_steps=MAX_STEPS,
        drone_start=DRONE_START,
        rx_positions=RX_POSITIONS if RX_POSITIONS else None,
        antenna_mode="SECTOR3_3GPP",  # "ISO", "SECTOR3_3GPP"
    )
    obs, info = env.reset(seed=0)
    done, trunc = False, False

    count = 0
    while not (done or trunc):
        a = [0, 0, 0]
        obs, rew, done, trunc, info = env.step(a)

        #time.sleep(1)  # para ver mejor la animación

        

        
    #time.sleep(20)  # para ver mejor la animación
    env.close()




"""
Métrica                                                   |  Significado y utilidad                                                                                                       
----------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------
SINR efectivo (sinr_eff_db)                               |  Relación señal a interferencia y ruido, medida efectiva tras procesamiento MIMO y precoding. Indicador de calidad del enlace.
Eficiencia espectral (se_la)                              |  Número de bits transmitidos con éxito por segundo y por Hertz. Mide cuánto se aprovecha el espectro radioeléctrico.          
Eficiencia espectral teórica (se_shannon)                 |  Límite superior teórico basado en el teorema de Shannon, sirve como referencia para evaluar desempeño real.                  
Índice MCS (mcs_index)                                    |  Modo de Modulación y Codificación seleccionado adaptativamente según calidad del canal. Indica tasa y robustez usada.        
Tasa de bits decodificados (num_decoded_bits)             |  Cantidad de bits correctamente recibidos luego de decodificación y corrección de errores.                                    
Feedback HARQ (harq_feedback)                             |  Indica si el bloque fue decodificado correctamente o necesita retransmisión. Fundamental para adaptación del enlace.         
Potencia transmitida ajustada (tx_power_per_ut)           |  Potencia que se asigna a cada usuario ajustada para control de potencia y fairness.                                          
Scheduling (is_scheduled y ut_scheduled)                  |  Asignación dinámica de recursos a usuarios, basado en políticas fair y tasas estimadas.                                      
Tasa o cantidad de recursos asignados (num_allocated_re)  |  Número de recursos de frecuencia-tiempo asignados efectivamente a cada usuario.                                              
Métrica de scheduling (pf_metric)                         |  Medida interna utilizada por el scheduler para balancear rendimiento y fairness entre usuarios.                              
Canales precodificados (h_eff)                            |  Canal tras aplicar precoding para MIMO, base para cálculo de SINR post igualación.                                           


"""

"""
SCENE = "santiago.xml"  # santiago.xml municg - san_francisco - simple_street_canyon - simple_street_canyon_with_cars
DRONE_START = (0.0, 0.0, 150.0)    # (x, y, z) en metros
RX_POSITIONS = [
    (-50.0, 0.0, 1.5),
    (-10.0,   260.0, 1.5),
    ( 60.0,  -30.0, 1.5),
    (50.0,   -160.0, 1.5),
    (  -345.0,    -272.0, 1.5),
    (277, 265, 1.5),
]
MAX_STEPS = 100
"""
