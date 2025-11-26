import matplotlib.pyplot as plt
import numpy as np
import sys
from pathlib import Path
from datetime import datetime

# --- CORRECCIÓN DE RUTA ---
# Usamos parents[2] para subir hasta 'Environment-drones'
# (igual que en run_sfm_test_v2.py)
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from env.environment.sionnaEnv import SionnaRT


def debug_scene_obstacles_slicer():
    # --- CONFIGURACIÓN ---
    SCENE_NAME = "simple_street_canyon_with_cars"

    RUN_TAG = datetime.now().strftime("%Y%m%d-%H%M%S")

    # Ahora guardamos en 'Pruebas SFM Slicer' (en la raíz del proyecto)
    # Le añadimos el prefijo DEBUG a la carpeta para diferenciarla de las corridas completas
    OUT_DIR = project_root / "Pruebas SFM Slicer" / f"DEBUG_{SCENE_NAME}_{RUN_TAG}"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"--- DEPURANDO OBSTÁCULOS (MÉTODO SLICER) EN {SCENE_NAME} ---")
    print(f"[INFO] Carpeta de salida: {OUT_DIR}")

    # 1. Cargar escena
    rt = SionnaRT(scene_name=SCENE_NAME)
    rt.build_scene()

    # 2. Usar el nuevo método "Slicer"
    obstacles_list = rt.get_sfm_obstacles(grid_density=0.3)

    if not obstacles_list:
        print("No se detectaron obstáculos.")
        return

    # obstacles_list es [ array(N, 2) ]
    all_points = obstacles_list[0]
    print(f"Se generó una nube de {len(all_points)} puntos.")

    # 3. Graficar (Scatter Plot)
    fig, ax = plt.subplots(figsize=(12, 10))

    # Dibujar los puntos como un mapa de "muros"
    ax.scatter(all_points[:, 0], all_points[:, 1], s=1, c='black', marker='.', label='Obstáculos (Z=0.3m a 2.5m)')

    ax.set_xlim(-80, 80)
    ax.set_ylim(-60, 60)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_title(f"Escaneo de Obstáculos: {SCENE_NAME}\n(Lo que la API SocialForce 'sentirá')")
    ax.legend()

    # Guardar imagen
    output_img = OUT_DIR / "debug_slicer_output.png"
    plt.savefig(output_img, dpi=150)
    print(f"Imagen de depuración guardada en: {output_img}")

    plt.close(fig)


if __name__ == "__main__":
    debug_scene_obstacles_slicer()