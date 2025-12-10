import matplotlib.pyplot as plt
import numpy as np
import sys
from pathlib import Path
from datetime import datetime

# --- Configuración de rutas para los archivos---
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from env.environment.sionnaEnv import SionnaRT


def debug_scene_obstacles_slicer():

    #1.-Carga de escena
    SCENE_NAME = "munich"

    print(f"--- Depurando obstáculos (Auto-Scale) EN {SCENE_NAME} ---")

    #Cargar motor de Sionna
    rt = SionnaRT(scene_name=SCENE_NAME)
    rt.build_scene()

    #2.-Cálculo automático de densidad
    #Se obtienen las dimensiones reales del mapa
    bounds = rt.mi_scene.bbox()
    extent_x = bounds.max.x - bounds.min.x
    extent_y = bounds.max.y - bounds.min.y
    max_dimension = max(extent_x, extent_y)

    print(f"[INFO] Dimensiones de la escena: {extent_x:.1f}m x {extent_y:.1f}m")

    #Lógica de Auto-Escalado (Auto-Scale)
    #Si el mapa es gigante (>500m), se usa menos resolución para no explotar la memoria.
    #Si es pequeño, se usa alta precisión.
    if max_dimension > 1000.0:
        auto_density = 1.5  #Muy grande
    elif max_dimension > 500.0:
        auto_density = 0.8  #Mediano-Grande
    else:
        auto_density = 0.3  #Pequeño/Estándar

    print(f"[INFO] Densidad calculada automáticamente: {auto_density}m")

    #3.-Ejecución del Slicer
    obstacles_list = rt.get_sfm_obstacles(grid_density=auto_density)

    if not obstacles_list:
        print("No se detectaron obstáculos.")
        return

    all_points = obstacles_list[0]
    print(f"Se generó una nube de {len(all_points)} puntos.")

    #4.-Graficación Dinámica
    #Se configuran las carpetas
    RUN_TAG = datetime.now().strftime("%Y%m%d-%H%M%S")
    OUT_DIR = project_root / "Pruebas SFM Slicer" / f"DEBUG_{RUN_TAG}_{SCENE_NAME}"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 12))

    #Se ajusta el tamaño del punto según la cantidad para que no se vea una mancha
    #A más puntos, menor el tamaño de estos
    marker_size = 10000 / len(all_points)
    marker_size = max(0.1, min(marker_size, 2.0))  #Limitar entre 0.1 y 2.0

    ax.scatter(all_points[:, 0], all_points[:, 1], s=marker_size, c='black', marker='.', label='Obstáculos')

    #Se usan los límites reales de la escena con un pequeño margen del 5%
    margin_x = extent_x * 0.05
    margin_y = extent_y * 0.05

    ax.set_xlim(bounds.min.x - margin_x, bounds.max.x + margin_x)
    ax.set_ylim(bounds.min.y - margin_y, bounds.max.y + margin_y)

    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_title(f"Escaneo: {SCENE_NAME}\nDimensión: {max_dimension:.0f}m | Resolución: {auto_density}m")

    output_img = OUT_DIR / f"debug_slicer_{SCENE_NAME}.png"
    plt.savefig(output_img, dpi=150)
    print(f"Imagen guardada en: {output_img}")
    plt.close(fig)


if __name__ == "__main__":
    debug_scene_obstacles_slicer()