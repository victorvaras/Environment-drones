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
from matplotlib.lines import Line2D
from datetime import datetime
from env.environment.gymnasium_env import DroneEnv  # <- usa tu archivo renombrado

import matplotlib.pyplot as plt




DEFAULTS_RT = {
    "max_depth": 5,
    "los": True,
    "specular_reflection": True,
    "diffuse_reflection": True,
    "refraction": True,
    "diffraction": True,
    "edge_diffraction": True,
    "diffraction_lit_region": True,
    "synthetic_array": False,
    "samples_per_src": 1_000_000,
    "max_num_paths_per_src": 1_000_000,
}

# Mapea clave -> etiqueta “humana” para el sufijo del archivo
LABELS_RT = {
    "max_depth": "max_depth",
    "los": "los",
    "specular_reflection": "specular_reflection",
    "diffuse_reflection": "diffuse_reflection",
    "refraction": "refraction",
    "diffraction": "diffraction",
    "edge_diffraction": "edge_diffraction",
    "diffraction_lit_region": "diffraction_lit_region",
    "synthetic_array": "synthetic_array",
    "samples_per_src": "samples_per_src",
    "max_num_paths_per_src": "max_num_paths_per_src",
}

def _bool_on_off(x: bool) -> str:
    return "ON" if bool(x) else "OFF"

def build_diff_suffix(rt_params: dict) -> str:
    """Crea sufijo de archivo con SOLO variables distintas a DEFAULTS_RT."""
    diffs = []
    for k, default_val in DEFAULTS_RT.items():
        cur_val = rt_params.get(k, None)
        if cur_val is None:
            continue
        if cur_val != default_val:
            key_short = LABELS_RT.get(k, k)
            if isinstance(cur_val, (bool, np.bool_)):
                diffs.append(f"{key_short}={_bool_on_off(cur_val)}")
            else:
                diffs.append(f"{key_short}={cur_val}")
    if not diffs:
        return ""
    return "__" + "__".join(str(d).replace(" ", "") for d in diffs)

def make_timing_plot_with_config(
    dt_list,
    rt_params: dict,
    out_dir: str | Path = "outputs",
    base_name: str = "tiempo_de_ejecucion",
    title: str | None = None,
):
    """
    Genera PNG con:
      - tiempo por step (s) en eje Y lineal,
      - tiempo acumulado (s) en eje secundario,
      - leyenda abajo (fuera) y en la misma leyenda se muestran los promedios,
      - SIN líneas horizontales de promedio.
    Nombre: {base_name}{diff_suffix}.png (solo difs vs DEFAULTS_RT).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    diff_suffix = build_diff_suffix(rt_params)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = out_dir / f"{ts}_{base_name}{diff_suffix}.png"

    dt = np.asarray(dt_list, dtype=float)
    steps = np.arange(len(dt), dtype=int)
    cum = np.cumsum(dt) if dt.size else np.array([])

    avg_from0 = float(np.mean(dt)) if dt.size > 0 else float("nan")
    avg_from1 = float(np.mean(dt[1:])) if dt.size > 1 else float("nan")
    cum_final = float(cum[-1]) if cum.size > 0 else float("nan")

    # --- Figura única ---
    fig = plt.figure(figsize=(12, 6))
    ax = plt.gca()
    ax2 = ax.twinx()

    # Curvas principales
    ax.plot(steps, dt, marker="o", linewidth=1.5, label="Tiempo por step (s)")
    ax2.plot(steps, cum, linestyle="--", linewidth=1.5, label="Tiempo acumulado (s)")

    # Etiquetas/título
    ax.set_xlabel("Step")
    ax.set_ylabel("Tiempo por step (s)")
    ax2.set_ylabel("Tiempo acumulado (s)")
    ax.set_title(title or "Tiempo por step y acumulado")

    # --- Leyenda combinada ABAJO (fuera) ---
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()

    # Entradas “de texto” para los promedios y acumulado final, sin línea ni marcador
    text_entries = []
    if np.isfinite(avg_from0):
        text_entries.append(Line2D([], [], linestyle="None",
                                   label=f"Promedio (desde step 0): {avg_from0:.6f} s"))
    if np.isfinite(avg_from1):
        text_entries.append(Line2D([], [], linestyle="None",
                                   label=f"Promedio (desde step 1): {avg_from1:.6f} s"))
    if np.isfinite(cum_final):
        text_entries.append(Line2D([], [], linestyle="None",
                                   label=f"Tiempo acumulado final: {cum_final:.6f} s"))

    legend = ax.legend(
        h1 + h2 + text_entries,
        l1 + l2 + [te.get_label() for te in text_entries],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=2,
        frameon=True,
        borderaxespad=0.0,
        handlelength=2.0
    )

    # Deja espacio inferior para la leyenda externa
    fig.subplots_adjust(bottom=0.28)

    ax.grid(True, linestyle=":", linewidth=0.8)
    fig.tight_layout()

    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    return str(out_path), avg_from0, avg_from1


OUT_DIR = Path.cwd() / "Environment drones" / "outputs-tiempos"
OUT_DIR.mkdir(parents=True, exist_ok=True)


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
MAX_STEPS = 100


if __name__ == "__main__":

    env = DroneEnv(
        scene_name=SCENE,
        max_steps=MAX_STEPS,
        drone_start=DRONE_START,
        rx_positions=RX_POSITIONS if RX_POSITIONS else None,
        antenna_mode="SECTOR3_3GPP",  # "ISO" o "SECTOR3_3GPP"
    )

    max_depth = env.rt.max_depth
    los = env.rt.los
    specular_reflection = env.rt.specular_reflection
    diffuse_reflection = env.rt.diffuse_reflection
    refraction = env.rt.refraction
    diffraction = env.rt.diffraction
    edge_diffraction = env.rt.edge_diffraction
    diffraction_lit_region = env.rt.diffraction_lit_region
    synthetic_array = env.rt.synthetic_array
    samples_per_src = env.rt.samples_per_src
    max_num_paths_per_src = env.rt.max_num_paths_per_src


    start_time = perf_counter()
    obs, info = env.reset(seed=0)
    done, trunc = False, False

    dt_list = []
    step_idx = 0
    while not (done or trunc):
        a = [0, 0, 0]
        b = [0, 0, 0]
        t0 = perf_counter()
        obs, rew, done, trunc, info = env.step(a, b)
        dt = perf_counter() - t0
        dt_list.append(dt)
        step_idx += 1


    # 2) RT params actuales
    rt_params = {
        "max_depth": env.rt.max_depth,
        "los": env.rt.los,
        "specular_reflection": env.rt.specular_reflection,
        "diffuse_reflection": env.rt.diffuse_reflection,
        "refraction": env.rt.refraction,
        "diffraction": env.rt.diffraction,
        "edge_diffraction": env.rt.edge_diffraction,
        "diffraction_lit_region": env.rt.diffraction_lit_region,
        "synthetic_array": env.rt.synthetic_array,
        "samples_per_src": env.rt.samples_per_src,
        "max_num_paths_per_src": env.rt.max_num_paths_per_src,
    }

    # 3) Generar PNG con nombre dinámico
    png_path, avg0, avg1 = make_timing_plot_with_config(
        dt_list,
        rt_params,
        out_dir=OUT_DIR,  # se crea si no existe
        base_name="tiempo_de_ejecucion",
        title="Tiempo por step y acumulado",
    )
    print("PNG guardado en:", png_path)
    print(f"Promedio desde step 0: {avg0:.6f} s")
    print(f"Promedio desde step 1: {avg1:.6f} s")






    end_time = perf_counter()
    elapsed = end_time - start_time
    print(f"Tiempo total episodio, 100 steps: 500mil rayos {elapsed:.3f} s")


    env.close()

