# run_sfm_test.py
# -*- coding: utf-8 -*-
import os
import sys
from pathlib import Path
import numpy as np
import matplotlib
from datetime import datetime

# Backend no interactivo
os.environ["MPLBACKEND"] = "Agg"
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# === Bootstrap sys.path ===
# Añade la raíz del proyecto al path para poder importar 'env'
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from env.environment.gymnasium_env import DroneEnv

# ========= Configuración =========
SCENE = "simple_street_canyon_with_cars"
DRONE_START = (0.0, 0.0, 10.0)

# Carpeta de salida con timestamp
RUN_TAG = datetime.now().strftime("%Y%m%d-%H%M%S")
#OUT_DIR = Path(project_root) / "outputs" / f"compare_metrics_{RUN_TAG}"
OUT_DIR = Path(project_root) / "Pruebas SFM Fast" / f"SFM_{RUN_TAG}"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --- DEFINICIÓN DE ESCENARIOS DE PRUEBA ---
RX_POSITIONS = [
    #(-50.0, 0.0, 1.5),
    #(20.0, -30.0, 1.5),
    #(20.0, 0.0, 1.5),
    #(-20.0, 0.0, 1.5),
    #(0, 0, 1.5),
    #(-1.0, 0.0, 1.5),
    #(0.0,   30.0, 1.5),
    #(20.0,  -30.0, 1.5),
    #(80.0,   40.0, 1.5),

    #Prueba original con 3 receptores
    #(20.0, -30.0, 1.5),
    #(50.0,    0.0, 1.5),
    #(-90, -55, 1.5),

    #Prueba de Colisión entre 2 receptores (sin colisión)
    #(-40.0, 0.0, 1.5),  #UE0
    #(-20.0, 0.0, 1.5),  #UE1

    #Prueba de Colisión entre 3 receptores (sin colisión)
    #(-40.0, 0.0, 1.5),   # UE0 empieza a la izquierda
    #(0.0, 0.0, 1.5),     # UE1 empieza en el centro
    #(-40.0, -10.0, 1.5), # UE2 empieza abajo a la izquierda

    #Prueba de Colisión entre 7 receptores (sin colisión)
    #Grupo 1 (4 agentes) - Empiezan a la izquierda
    (-40.0, 1.0, 1.5),  #UE0
    (-40.0, -1.0, 1.5), #UE1
    (-38.0, 0.5, 1.5),  #UE2
    (-38.0, -0.5, 1.5), #UE3

    # Grupo 2 (3 agentes) - Empiezan a la derecha
    (-10.0, 0.0, 1.5),  #UE4
    (-10.0, 1.0, 1.5),  #UE5
    (-10.0, -1.0, 1.5), #UE6
]

RX_GOALS = [
    #Prueba original con 3 receptores
    #(80.0, -30.0, 1.5),  # Meta UE0 (20.0, -30.0, 1.5)
    #(-80.0, 0.0, 1.5),   # Meta UE1 (50.0,   0.0, 1.5)
    #(80.0, -55.0, 1.5),  # Meta UE2 (-90,    -55, 1.5)

    #Prueba de Colisión entre 2 receptores
    #(-10.0, 0.0, 1.5),   # UE0 quiere ir a la derecha
    #(-50.0, 0.0, 1.5),  # UE1 quiere ir a la izquierda

    #Prueba de Colisión entre 3 receptores
    #(0.0, 0.0, 1.5),     # UE0 (Verde) quiere ir al centro
    #(-40.0, 0.0, 1.5),   # UE1 (Naranjo) quiere ir a la izquierda
    #(-20.0, -5.0, 1.5),  # UE2 (Rojo) quiere ir en diagonal

    #Prueba de Colisión entre 7 receptores (sin colisión)
    # Metas Grupo 1
    (-10.0, 1.0, 1.5),   # UE0
    (-10.0, -1.0, 1.5),  # UE1
    (-12.0, 0.5, 1.5),   # UE2
    (-12.0, -0.5, 1.5),  # UE3

    # Metas Grupo 2
    (-40.0, 0.0, 1.5),   # UE4
    (-40.0, 1.0, 1.5),   # UE5
    (-40.0, -1.0, 1.5),  # UE6
]

MAX_STEPS = 300
FREQS_MHZ = [3500.0]


def _scalar_float(x):
    """Convierte x (Dr.Jit Float, numpy scalar, etc.) a float de Python."""
    try:
        return float(x)
    except:
        pass
    try:
        return np.float64(x).item()
    except:
        pass
    if hasattr(x, "value"): return float(x.value)
    return 0.0  # Fallback


def _vec3_to_np(p):
    """Convierte un vector posición a np.array(3,), dtype=float."""
    if isinstance(p, (list, tuple, np.ndarray)) and len(p) == 3:
        return np.array([_scalar_float(p[0]), _scalar_float(p[1]), _scalar_float(p[2])], dtype=float)
    if hasattr(p, "x") and hasattr(p, "y") and hasattr(p, "z"):
        return np.array([_scalar_float(p.x), _scalar_float(p.y), _scalar_float(p.z)], dtype=float)
    return np.array([0., 0., 0.])


def _get_drone_xyz(rt):
    if hasattr(rt, "tx") and rt.tx: return _vec3_to_np(rt.tx.position)
    if hasattr(rt, "txs") and rt.txs: return _vec3_to_np(rt.txs[0].position)
    return np.array([0., 0., 10.])


def _get_rx_positions_xyz(rt):
    pos = []
    for rx in rt.rx_list:
        pos.append(_vec3_to_np(rx.position))
    return np.vstack(pos).astype(float)


# --- Plot de las  trayectorias ---
def plot_trajectories_xy_xz(tracks: dict, out_path: Path, title_prefix="Trayectorias",
                            step_stride=5, show_step_labels=False, rx_labels=None):
    drone = tracks["drone"];
    ues = tracks["ues"];
    steps = tracks["steps"]
    T = drone.shape[0];
    N = ues.shape[1]
    if rx_labels is None: rx_labels = [f"UE{i}" for i in range(N)]

    def _auto_limits_xy(pad=5.0):
        d = drone[:, :2];
        u = ues[:, :, :2].reshape(-1, 2)
        allxy = np.vstack([d, u])
        # Manejo de nans por si acaso
        return (np.nanmin(allxy, axis=0)[0] - pad, np.nanmax(allxy, axis=0)[0] + pad,
                np.nanmin(allxy, axis=0)[1] - pad, np.nanmax(allxy, axis=0)[1] + pad)

    xmin, xmax, ymin, ymax = _auto_limits_xy(pad=5.0)
    fig = plt.figure(figsize=(13.8, 7.6))
    gs = GridSpec(2, 1, height_ratios=[2, 1], figure=fig)
    fig.subplots_adjust(left=0.08, right=0.80, top=0.90, bottom=0.08, hspace=0.32)
    ax_xy = fig.add_subplot(gs[0, 0]);
    ax_xz = fig.add_subplot(gs[1, 0])

    # XY
    ax_xy.set_title(f"{title_prefix} — Planta (XY)", pad=12)
    ax_xy.set_xlabel("X [m]");
    ax_xy.set_ylabel("Y [m]")
    ax_xy.set_aspect("equal", adjustable="box")
    ax_xy.set_xlim([xmin, xmax]);
    ax_xy.set_ylim([ymin, ymax])
    ax_xy.grid(True, ls="--", alpha=0.35);
    ax_xy.set_axisbelow(True)

    ax_xy.plot(drone[:, 0], drone[:, 1], lw=2.0, label="Drone", zorder=2.5)
    ax_xy.scatter(drone[0, 0], drone[0, 1], s=110, marker="^", label="Drone start", zorder=3)
    ax_xy.scatter(drone[-1, 0], drone[-1, 1], s=110, marker=">", label="Drone end", zorder=3)

    for i in range(N):
        traj = ues[:, i, :]
        ax_xy.plot(traj[:, 0], traj[:, 1], lw=1.8, label=rx_labels[i], zorder=2.2)
        ax_xy.scatter(traj[0, 0], traj[0, 1], s=70, marker="o", zorder=2.8)
        ax_xy.scatter(traj[-1, 0], traj[-1, 1], s=70, marker="s", zorder=2.8)
        ax_xy.text(traj[0, 0] + 0.6, traj[0, 1] + 0.6, rx_labels[i], fontsize=9, weight="bold", zorder=3)

        for k in range(0, T - 1, step_stride):
            dx = traj[k + 1, 0] - traj[k, 0];
            dy = traj[k + 1, 1] - traj[k, 1]
            ax_xy.arrow(traj[k, 0], traj[k, 1], dx, dy, head_width=0.6, alpha=0.6, zorder=2.4)

    # XZ
    ax_xz.set_title("Perfil XZ", pad=10)
    ax_xz.set_xlabel("X [m]");
    ax_xz.set_ylabel("Z [m]")
    ax_xz.grid(True, ls="--", alpha=0.35)
    ax_xz.plot(drone[:, 0], drone[:, 2], lw=2.0, label="Drone")
    ax_xz.scatter(drone[0, 0], drone[0, 2], s=90, marker="^")
    for i in range(N):
        traj = ues[:, i, :]
        ax_xz.plot(traj[:, 0], traj[:, 2], lw=1.4, label=rx_labels[i])
        ax_xz.scatter(traj[0, 0], traj[0, 2], s=60, marker="o")
        ax_xz.scatter(traj[-1, 0], traj[-1, 2], s=60, marker="s")
        ax_xz.text(traj[0, 0] + 0.6, traj[0, 2] + 0.6, rx_labels[i], fontsize=8.5)

    # Leyenda
    handles, labels = [], []
    for ax in (ax_xy, ax_xz):
        h, l = ax.get_legend_handles_labels()
        handles += h;
        labels += l
    seen = set();
    H = [];
    L = []
    for h, l in zip(handles, labels):
        if l not in seen: H.append(h); L.append(l); seen.add(l)
    fig.legend(H, L, loc="center left", bbox_to_anchor=(0.82, 0.5), title="Leyenda")

    fig.suptitle("Trayectorias — Drone y UEs", y=0.97, fontsize=14)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run_episode(freq_mhz: float) -> dict:
    print(f"--- Iniciando Episodio {freq_mhz} MHz ---")

    env = DroneEnv(
        render_mode=None,
        scene_name=SCENE,
        max_steps=MAX_STEPS,
        drone_start=DRONE_START,
        rx_positions=RX_POSITIONS,
        rx_goals=RX_GOALS,
        antenna_mode="SECTOR3_3GPP",
        frequency_mhz=freq_mhz,
        run_metrics=False
    )

    obs, info = env.reset(seed=0)
    drone_traj, ue_traj, steps = [], [], []

    # Guardar mapa de calor
    out_img = OUT_DIR / f"radio_map_{int(freq_mhz)}MHz.png"
    env.rt.render_scene_to_file(filename=str(out_img), with_radio_map=True)
    print("Mapa de calor guardado.")

    print("Simulando pasos...")
    t = 0
    # Bucle simple sin métricas
    while t < MAX_STEPS:
        # Snapshot de posiciones
        drone_traj.append(_get_drone_xyz(env.rt).copy())
        ue_traj.append(_get_rx_positions_xyz(env.rt).copy())
        steps.append(t)

        # Step del entorno (movimiento SFM)
        obs, rew, done, trunc, info = env.step(np.array([0, 0, 0], dtype=float))

        if t % 50 == 0: print(f"  Step {t}/{MAX_STEPS}")
        t += 1
        if done or trunc: break

    # Snapshot final
    drone_traj.append(_get_drone_xyz(env.rt).copy())
    ue_traj.append(_get_rx_positions_xyz(env.rt).copy())
    steps.append(t)
    env.close()

    return {
        "freq_mhz": freq_mhz,
        "tracks": {
            "drone": np.vstack(drone_traj),
            "ues": np.stack(ue_traj, axis=0),
            "steps": np.array(steps, dtype=int),
        },
    }


def main():
    print(f"[INFO] Guardando resultados RÁPIDOS en: {OUT_DIR}")

    for f in FREQS_MHZ:
        r = run_episode(f)

        out_traj = OUT_DIR / f"traj_SFM_FAST_{int(f)}MHz.png"
        title = f"TEST RÁPIDO SFM — {f:.0f} MHz"

        plot_trajectories_xy_xz(r["tracks"], out_path=out_traj,
                                title_prefix=title, step_stride=5)
        print(f"Gráfico guardado: {out_traj}")

    print(f"[DONE] ¡Listo!")


if __name__ == "__main__":
    main()