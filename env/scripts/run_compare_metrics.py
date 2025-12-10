# run_compare_metrics.py
# -*- coding: utf-8 -*-

# === Limpieza de logs===
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  #0 = all, 1 = info, 2 = warnings, 3 = solo errors

# === Bootstrap sys.path a la raíz del proyecto (dos niveles arriba) ===
import sys
import time
from pathlib import Path
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Backend no interactivo para guardar imágenes
import os
os.environ["MPLBACKEND"] = "Agg"

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.animation as animation
from matplotlib.ticker import ScalarFormatter, MaxNLocator
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

# Desplazamientos y estilos por frecuencia para evitar solapamiento
_X_OFFSETS = [-0.12, -0.04, 0.04, 0.12, 0.20, -0.20]  # se cicla si hay >6 freqs
_MARKERS   = ["o", "s", "D", "^", "v", "P"]
_LINESTY   = ["-", "--", "-.", ":", "-", "--"]



from env.environment.gymnasium_env import DroneEnv  # adapta si tu ruta difiere


# ========= Configuración =========
SCENE = "simple_street_canyon_with_cars"  # p.ej. "santiago.xml", "munich"
DRONE_START = (0.0, 0.0, 10.0)

SEMILLA = 0
#Semilla (seed) de la simulación
#Cantidad de agentes a generar aleatoriamente
NUM_AGENTS = 10
#Posiciones iniciales
RX_POSITIONS = None
#Metas de los receptores
RX_GOALS = None
#Máximo de pasos para la simulación (N° de steps de la simulación)
MAX_STEPS = 1000
#Frecuencias de la simulación
FREQS_MHZ = [3500.0]
FREQ_LABELS = [f"{f:.0f} MHz" for f in FREQS_MHZ]

# Carpeta de salida con timestamp
RUN_TAG = datetime.now().strftime("%Y%m%d-%H%M%S")
OUT_DIR = Path(project_root) / "Pruebas finales" / f"METRICS_{RUN_TAG}_{SCENE}_{NUM_AGENTS} agentes_{SEMILLA} (seed)_{MAX_STEPS} steps"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_DIR_RECEPTORS = OUT_DIR / "receptors-metrics"
OUT_DIR_RECEPTORS.mkdir(parents=True, exist_ok=True)

OUT_DIR_UE_METRICS = OUT_DIR / "metricas-por-usuario"
OUT_DIR_UE_METRICS.mkdir(parents=True, exist_ok=True)

OUT_DIR_UE_all_METRICS = OUT_DIR / "metricas-totales-por-usuario-y-frecuencia"
OUT_DIR_UE_all_METRICS.mkdir(parents=True, exist_ok=True)

OUT_DIR_FREQ_METRICS = OUT_DIR / "metricas-totales-por-frecuencia"
OUT_DIR_FREQ_METRICS.mkdir(parents=True, exist_ok=True)

OUT_DIR_DOPPLER = OUT_DIR / "doppler-metrics"
OUT_DIR_DOPPLER.mkdir(parents=True, exist_ok=True)


def _scalar_float(x):
    """Convierte x (Dr.Jit Float, numpy scalar, etc.) a float de Python."""
    # 1) intento directo
    try:
        return float(x)
    except Exception:
        pass
    # 2) numpy intenta sacar escalar
    try:
        a = np.asarray(x)
        if a.shape == ():  # escalar numpy
            return float(a)
    except Exception:
        pass
    # 3) atributo 'value' (algunos wrappers)
    if hasattr(x, "value"):
        try:
            return float(x.value)
        except Exception:
            pass
    # 4) último intento: castear a np.float64
    try:
        return np.float64(x).item()
    except Exception:
        raise TypeError(f"No pude convertir a float el objeto de tipo {type(x)}: {x!r}")


def _vec3_to_np(p):
    """Convierte un vector posición cualquiera (Vector3f, iterable, etc.) a np.array(3,), dtype=float."""
    # Caso list/tuple/np con longitud 3
    if isinstance(p, (list, tuple, np.ndarray)) and len(p) == 3:
        return np.array([_scalar_float(p[0]), _scalar_float(p[1]), _scalar_float(p[2])], dtype=float)

    # Caso objetos tipo Mitsuba/DrJit con atributos x,y,z
    if hasattr(p, "x") and hasattr(p, "y") and hasattr(p, "z"):
        return np.array([_scalar_float(p.x), _scalar_float(p.y), _scalar_float(p.z)], dtype=float)

    # Caso iterable genérico (e.g., Vector3f iterable)
    try:
        lst = list(p)
        if len(lst) >= 3:
            return np.array([_scalar_float(lst[0]), _scalar_float(lst[1]), _scalar_float(lst[2])], dtype=float)
    except Exception:
        pass

    # Intento numpy plano
    try:
        arr = np.asarray(p, dtype=float).reshape(-1)
        if arr.size >= 3:
            return arr[:3]
    except Exception:
        pass

    raise TypeError(f"No pude extraer un vec3 de tipo {type(p)} ({p!r}).")


def _get_drone_xyz(rt) -> np.ndarray:
    """Posición del 'drone' (tu TX único) como np.array([x,y,z], float)."""
    # Variante antigua: dron.pos
    if hasattr(rt, "dron") and hasattr(rt.dron, "pos"):
        return _vec3_to_np(rt.dron.pos)

    # Implementación actual: self.tx o self.txs[0]
    if hasattr(rt, "tx") and hasattr(rt.tx, "position"):
        return _vec3_to_np(rt.tx.position)

    if hasattr(rt, "txs") and isinstance(rt.txs, (list, tuple)) and rt.txs:
        tx0 = rt.txs[0]
        if hasattr(tx0, "position"):
            return _vec3_to_np(tx0.position)

    raise AttributeError("No pude obtener la posición del dron/tx (faltan atributos esperados).")


def _get_rx_positions_xyz(rt) -> np.ndarray:
    """Posiciones de todos los RX como np.array shape (N,3), float."""
    # Tu implementación actual: lista de Receiver
    if hasattr(rt, "rx_list") and isinstance(rt.rx_list, (list, tuple)) and rt.rx_list:
        pos = []
        for rx in rt.rx_list:
            if hasattr(rx, "position"):
                pos.append(_vec3_to_np(rx.position))
            else:
                raise AttributeError("Un RX en rx_list no tiene atributo 'position'.")
        return np.vstack(pos).astype(float)

    # Variante anterior:
    if hasattr(rt, "receptores") and hasattr(rt.receptores, "positions_xyz"):
        # Nota: si esto devuelve tipos DrJit, _vec3_to_np lo maneja igual
        arr = rt.receptores.positions_xyz()
        # normalizamos a (N,3) float
        arr = np.asarray(arr)
        out = []
        for i in range(arr.shape[0]):
            out.append(_vec3_to_np(arr[i]))
        return np.vstack(out).astype(float)

    raise AttributeError("No pude obtener posiciones de receptores (rx_list o receptores.positions_xyz).")


# === 1. PLOT ESTÁTICO (FINAL: Con etiquetas de texto) ===
def plot_trajectories_xy_xz(tracks, obstacles, scene_bounds, out_path, freq_mhz, num_agents, seed):
    drone = tracks["drone"]
    ues = tracks["ues"]
    T = ues.shape[0]
    N = ues.shape[1]
    rx_labels = [f"UE{i}" for i in range(N)]

    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(2, 1, height_ratios=[3, 1.2], figure=fig)
    fig.subplots_adjust(left=0.08, right=0.82, top=0.92, bottom=0.08, hspace=0.35)

    ax_xy = fig.add_subplot(gs[0, 0])
    ax_xz = fig.add_subplot(gs[1, 0])

    # --- PLANTA (XY) ---
    ax_xy.set_title(f"Escenario: {SCENE}\nFrecuencia: {freq_mhz:.0f} MHz | {num_agents} Agentes | Semilla N° {seed} ({T} pasos)",
                    pad=12, fontsize=14, weight='bold')
    ax_xy.set_xlabel("X [m]", fontsize=12)
    ax_xy.set_ylabel("Y [m]", fontsize=12)
    ax_xy.set_aspect("equal", adjustable='datalim')

    # LÍMITES
    if scene_bounds:
        (xmin, xmax) = scene_bounds[0]
        (ymin, ymax) = scene_bounds[1]
        extent_x = xmax - xmin
        extent_y = ymax - ymin
        margin_x = extent_x * 0.05
        margin_y = extent_y * 0.05
        ax_xy.set_xlim(xmin - margin_x, xmax + margin_x)
        ax_xy.set_ylim(ymin - margin_y, ymax + margin_y)
    else:
        all_xy = np.vstack([drone[:, :2], ues[:, :, :2].reshape(-1, 2)])
        pad = 10
        ax_xy.set_xlim(np.min(all_xy[:, 0]) - pad, np.max(all_xy[:, 0]) + pad)
        ax_xy.set_ylim(np.min(all_xy[:, 1]) - pad, np.max(all_xy[:, 1]) + pad)

    ax_xy.grid(True, ls="--", alpha=0.35)

    # --- OBSTÁCULOS (CAMBIO: MÁS OSCUROS) ---
    if obstacles:
        obs_stack = np.vstack(obstacles)
        n_points = len(obs_stack)
        if n_points > 0:
            marker_size = 10000.0 / n_points
            marker_size = max(0.1, min(marker_size, 2.0))

            ax_xy.scatter(obs_stack[:, 0], obs_stack[:, 1],
                       s=marker_size, c='black', marker='.', alpha=1.0)

    # Dron XY
    ax_xy.plot(drone[:, 0], drone[:, 1], lw=3.0, label="Drone", zorder=2.5, color='tab:blue')
    ax_xy.scatter(drone[0, 0], drone[0, 1], s=150, marker="^", label="Drone start", zorder=3, edgecolors='k',
                  color='tab:blue')

    # UEs XY
    colors = matplotlib.colormaps['tab10']
    for i in range(N):
        c = colors(i % 10)
        traj = ues[:, i, :]
        ax_xy.plot(traj[:, 0], traj[:, 1], lw=3, color=c, label=rx_labels[i], zorder=2.2, alpha=0.9)
        ax_xy.scatter(traj[0, 0], traj[0, 1], s=120, marker="o", color=c, zorder=2.8, edgecolors='white')
        ax_xy.scatter(traj[-1, 0], traj[-1, 1], s=120, marker="s", color=c, zorder=2.8, edgecolors='white')

        if N <= 20:
            ax_xy.text(traj[0, 0] + 1.5, traj[0, 1] + 1.5, rx_labels[i], fontsize=11, weight="bold", color=c, zorder=5)

    # --- PERFIL (XZ) (CAMBIO: INICIO CÍRCULO / FIN CUADRADO) ---
    ax_xz.set_title("Perfil de Elevación (XZ)", pad=10, fontsize=14, weight='bold')
    ax_xz.set_xlabel("X [m]", fontsize=12)
    ax_xz.set_ylabel("Z [m]", fontsize=12)
    ax_xz.grid(True, ls="--", alpha=0.35)

    ax_xz.set_xlim(ax_xy.get_xlim())
    ax_xz.set_ylim(0, 18)
    ax_xz.set_aspect('auto')

    # Dron
    ax_xz.plot(drone[:, 0], drone[:, 2], lw=3.0, label="Drone", color='tab:blue')
    ax_xz.scatter(drone[0, 0], drone[0, 2], s=100, marker="^", color='tab:blue', edgecolors='k')

    # UEs
    for i in range(N):
        c = colors(i % 10)
        # Trayectoria
        ax_xz.plot(ues[:, i, 0], ues[:, i, 2], lw=2.5, color=c, label=rx_labels[i], alpha=0.8)
        # Inicio (Círculo)
        ax_xz.scatter(ues[0, i, 0], ues[0, i, 2], s=80, marker="o", color=c, edgecolors='white', zorder=3)
        # Fin (Cuadrado)
        ax_xz.scatter(ues[-1, i, 0], ues[-1, i, 2], s=80, marker="s", color=c, edgecolors='white', zorder=3)

    # Leyenda
    obst_proxy = Line2D([0], [0], marker='o', color='w', label='Obstáculos', markerfacecolor='#202020', markersize=8,
                        alpha=0.6)
    handles_xy, labels_xy = ax_xy.get_legend_handles_labels()
    unique_labels = dict(zip(labels_xy, handles_xy))

    if N > 15:
        final_handles = [unique_labels['Drone']] + [obst_proxy]
        final_labels = ['Drone', 'Obstáculos']
    else:
        final_handles = [unique_labels['Drone']] + [unique_labels[l] for l in rx_labels if l in unique_labels] + [
            obst_proxy]
        final_labels = ['Drone'] + [l for l in rx_labels if l in unique_labels] + ['Obstáculos']

    fig.legend(final_handles, final_labels, loc="center left", bbox_to_anchor=(0.83, 0.5), title="Leyenda")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# === 2. GENERADOR DE GIF (CON LEYENDA EXTERNA) ===
def make_gif(tracks, obstacles, scene_bounds, out_path, fps=20):
    ues = tracks["ues"]
    T, N, _ = ues.shape

    fig, ax = plt.subplots(figsize=(12, 12))
    fig.subplots_adjust(right=0.80)

    ax.set_aspect('equal')
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_title(f"Simulación Dinámica (Sionna + SocialForce)\nEscenario: {SCENE} | {N} Agentes | Semilla N° {SEMILLA} ({T} pasos)",
                 pad=12, fontsize=14, weight='bold')
    ax.grid(True, alpha=0.3)

    if obstacles:
        obs_stack = np.vstack(obstacles)
        n_points = len(obs_stack)
        if n_points > 0:
            marker_size = 10000.0 / n_points
            marker_size = max(0.1, min(marker_size, 2.0))

            # Igualamos alpha a 0.6
            ax.scatter(obs_stack[:, 0], obs_stack[:, 1],
                       s=marker_size, c='black', marker='.', alpha=1.0)

    if scene_bounds:
        (xmin, xmax) = scene_bounds[0]
        (ymin, ymax) = scene_bounds[1]
        extent_x = xmax - xmin
        extent_y = ymax - ymin
        margin_x = extent_x * 0.05
        margin_y = extent_y * 0.05
        ax.set_xlim(xmin - margin_x, xmax + margin_x)
        ax.set_ylim(ymin - margin_y, ymax + margin_y)
    else:
        all_xy = ues[:, :, :2].reshape(-1, 2)
        pad = 5
        ax.set_xlim(np.min(all_xy[:, 0]) - pad, np.max(all_xy[:, 0]) + pad)
        ax.set_ylim(np.min(all_xy[:, 1]) - pad, np.max(all_xy[:, 1]) + pad)

    cmap = matplotlib.colormaps['tab10']
    colors = [cmap(i % 10) for i in range(N)]

    start_pos = ues[0, :, :2]
    scats = ax.scatter(start_pos[:, 0], start_pos[:, 1], s=100, c=colors, zorder=5, edgecolors='white')
    trails = [ax.plot([], [], '-', lw=2, color=colors[i], alpha=0.6)[0] for i in range(N)]

    # Leyenda GIF
    if N <= 15:
        custom_lines = [Line2D([0], [0], color=colors[i], lw=2) for i in range(N)]
        ax.legend(custom_lines, [f"UE{i}" for i in range(N)],
                  loc='center left', bbox_to_anchor=(1.02, 0.5), title="Receptores")
    else:
        ax.legend([scats], [f"{N} Agentes"], loc='center left', bbox_to_anchor=(1.02, 0.5))

    def update(frame):
        current_pos = ues[frame, :, :2]
        scats.set_offsets(current_pos)
        start = max(0, frame - 30)
        for i, trail in enumerate(trails):
            trail.set_data(ues[start:frame + 1, i, 0], ues[start:frame + 1, i, 1])
        return scats, *trails

    # --- LÓGICA DE SEGURIDAD ADAPTATIVA ---
    # Si T es pequeño (ej: 300), step_skip será 1 (sin saltos, fluido).
    # Si T es gigante (ej: 10000), step_skip será alto (ej: 20) para no explotar.
    TARGET_TOTAL_FRAMES = 250

    if T <= TARGET_TOTAL_FRAMES:
        step_skip = 1  # Muestra TODOS los cuadros (Perfecto para escena simple)
    else:
        step_skip = T // TARGET_TOTAL_FRAMES  # Salta para proteger RAM

    print(f"[GIF] Generando animación con step_skip = {step_skip} (Total frames: {T // step_skip})")

    frames_idx = range(0, T, step_skip)

    ani = animation.FuncAnimation(fig, update, frames=frames_idx, interval=50, blit=True)
    ani.save(out_path, writer='pillow', fps=fps)
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
        num_agents=NUM_AGENTS,
        antenna_mode="SECTOR3_3GPP",
        frequency_mhz=freq_mhz,
        run_metrics=True
    )

    # Recuperar límites
    scene_bounds = env.scene_bounds

    # Reset con Semilla
    obs, info = env.reset(seed=SEMILLA)
    done = trunc = False

    # Extraer obstáculos para visualización
    try:
        if hasattr(env, "mobility_manager") and env.mobility_manager.sfm_sim:
            sfm_obstacles_torch = env.mobility_manager.sfm_sim.ped_space.space
        elif hasattr(env, "sfm_sim") and env.sfm_sim:
            sfm_obstacles_torch = env.sfm_sim.ped_space.space
        else:
            sfm_obstacles_torch = []
        obstacles_np = [o.numpy() for o in sfm_obstacles_torch]
    except Exception:
        obstacles_np = []

    steps_ue_metrics, steps_tbler_running = [], []
    drone_traj, ue_traj, steps = [], [], []

   
    out_img = OUT_DIR / f"radio_map_{SCENE}_{NUM_AGENTS} agentes_{SEMILLA} (seed)_{MAX_STEPS} steps.png"
    env.rt.render_scene_to_file(filename=str(out_img), with_radio_map=True)

    t = 0
    while not (done or trunc):
        # Snapshot trayectoria
        drone_traj.append(_get_drone_xyz(env.rt).copy())
        ue_traj.append(_get_rx_positions_xyz(env.rt).copy())
        steps.append(t)

        a = [5, 0, 0]
        b = [0, 0, 0]

        # Recolección Métricas
        ue_metrics_step = info.get("ue_metrics", [])
        tbler_running   = info.get("tbler_running_per_ue", None)

        # Guardado robusto (por si llega vacío)
        if ue_metrics_step:
            steps_ue_metrics.append([dict(m) for m in ue_metrics_step])
        else:
            steps_ue_metrics.append([])

        steps_tbler_running.append(list(tbler_running) if tbler_running is not None else [np.nan]*NUM_AGENTS)

        t += 1

    env.close()

    return {
        "freq_mhz": freq_mhz,
        "steps_ue_metrics": steps_ue_metrics,
        "steps_tbler_running": steps_tbler_running,
        "obstacles": obstacles_np,
        "bounds": scene_bounds,
        "tracks": {
            "drone": np.vstack(drone_traj),
            "ues": np.stack(ue_traj, axis=0),
            "steps": np.array(steps, dtype=int),
        },
    }


def to_dataframe(run_dict: dict) -> pd.DataFrame:
    """
    Convierte la corrida en un DataFrame con índice (step, ue_id)
    y columnas de métricas.
    """
    freq = run_dict["freq_mhz"]
    steps_ue_metrics = run_dict["steps_ue_metrics"]
    steps_tbler_running = run_dict["steps_tbler_running"]

    rows = []
    for t, (ue_list, tbler_run_vec) in enumerate(zip(steps_ue_metrics, steps_tbler_running)):
        # Asegura mismo largo entre ue_list y tbler_run_vec
        num_ut = max(len(ue_list), len( tbler_run_vec or [] ))
        for i in range(num_ut):
            # Saca del ue_list si existe; si no, NaNs
            if i < len(ue_list):
                m = ue_list[i]
                ue_id = m.get("ue_id", i)
                sinr = m.get("sinr_eff_db", np.nan)
                prx  = m.get("prx_dbm", np.nan)
                prx_theo = m.get("prx_dbm_theo", np.nan)
                se_la = m.get("se_la", np.nan)
                se_sh = m.get("se_shannon", np.nan)
                se_gap = m.get("se_gap_pct", np.nan)
                tbler_step = m.get("tbler", np.nan)

                doppler_fd_hz  = m.get("doppler_fd_hz", np.nan)
                doppler_slope  = m.get("doppler_slope_rad_per_sym", np.nan)
                doppler_nu     = m.get("doppler_nu_fd_over_scs", np.nan)
                doppler_Tc_sec = m.get("doppler_Tc_seconds", np.nan)
            else:
                ue_id = i
                sinr = prx = se_la = se_sh = se_gap = tbler_step = np.nan
                doppler_fd_hz = doppler_slope = doppler_nu = doppler_Tc_sec = np.nan

            # TBLER running
            tbler_run = tbler_run_vec[i] if (tbler_run_vec is not None and i < len(tbler_run_vec)) else np.nan

            rows.append({
                "freq_mhz": freq,
                "step": t + 1,  # 1-based para lectura
                "ue_id": ue_id,
                "sinr_eff_db": float(sinr) if sinr is not None else np.nan,
                "prx_dbm": float(prx) if prx is not None else np.nan,
                "prx_dbm_theo": float(prx_theo) if prx_theo is not None else np.nan,
                "se_la": float(se_la) if se_la is not None else np.nan,
                "se_shannon": float(se_sh) if se_sh is not None else np.nan,
                "se_gap_pct": float(se_gap) if se_gap is not None else np.nan,
                "tbler_step": float(tbler_step) if tbler_step is not None else np.nan,
                "tbler_running": float(tbler_run) if tbler_run is not None else np.nan,

                "doppler_fd_hz": float(doppler_fd_hz) if doppler_fd_hz is not None else np.nan,
                "doppler_slope_rad_per_sym": float(doppler_slope) if doppler_slope is not None else np.nan,
                "doppler_nu": float(doppler_nu) if doppler_nu is not None else np.nan,
                "doppler_Tc_ms": float(doppler_Tc_sec)*1e3 if doppler_Tc_sec not in (None, np.nan) else np.nan,
            })

    df = pd.DataFrame(rows)
    return df



def plot_metric_per_ue(df_all: pd.DataFrame, metric: str, ylabel: str, out_dir: Path,
                       legend_outside: bool = True):
    """
    Un PNG por UE para la métrica indicada (p. ej., 'prx_dbm', 'sinr_eff_db', etc.).
    Si metric == 'prx_dbm', se añade la curva teórica 'prx_dbm_theo' como cota superior.
    """
    ue_ids = sorted(df_all["ue_id"].dropna().astype(int).unique().tolist())
    freqs  = sorted(df_all["freq_mhz"].dropna().unique().tolist())
    label_for = {f: f"{f:.0f} MHz" for f in freqs}

    # Título legible por métrica
    nice_title = {
        "prx_dbm": "PRx (dBm)",
        "sinr_eff_db": "SINR (dB)",
        "se_la": "SE-LA (b/s/Hz)",
        "se_shannon": "SE Shannon (b/s/Hz)",
        "tbler_running": "TBLER running",
        "tbler_step": "TBLER (por step)",
    }.get(metric, metric)

    for ue in ue_ids:
        fig = plt.figure(figsize=(9.8, 5.6))
        ax = plt.gca()

        yvals_all = []

        for j, f in enumerate(freqs):
            dfx = df_all[(df_all["ue_id"] == ue) & (df_all["freq_mhz"] == f)].sort_values("step")
            x   = dfx["step"].to_numpy(dtype=float) + _X_OFFSETS[j % len(_X_OFFSETS)]
            y   = dfx[metric].to_numpy(dtype=float)

            # curva medida
            line_meas, = ax.plot(
                x, y,
                marker=_MARKERS[j % len(_MARKERS)],
                linewidth=1.8,
                linestyle=_LINESTY[j % len(_LINESTY)],
                markersize=5.5,
                label=label_for[f],
            )
            yvals_all.append(y[~np.isnan(y)])

            # --- cota teórica para PRx ---
            if metric == "prx_dbm" and "prx_dbm_theo" in dfx.columns:
                y_th = dfx["prx_dbm_theo"].to_numpy(dtype=float)
                # usa el mismo color que la curva medida para asociarlas visualmente
                color_meas = line_meas.get_color()
                ax.plot(
                    x, y_th,
                    linestyle=":",
                    linewidth=2.0,
                    color=color_meas,
                    label=f"{label_for[f]} · Teórico",
                )
                yvals_all.append(y_th[~np.isnan(y_th)])

        # títulos/etiquetas
        ax.set_title(f"{nice_title} por step — UE {ue}", pad=10)
        ax.set_xlabel("Step")
        ax.set_ylabel(ylabel)

        # formato dB/dBm si aplica
        if metric.lower().endswith("db") or metric in ("prx_dbm", "sinr_eff_db"):
            _axis_format_db(ax)
        else:
            ax.grid(True, linestyle="--", alpha=0.35)
            ax.set_axisbelow(True)

        # auto-límites con padding si hay datos
        flat = (np.concatenate([v for v in yvals_all if v.size > 0])
                if any(len(v) > 0 for v in yvals_all) else np.array([]))
        if flat.size:
            y_min, y_max = np.nanmin(flat), np.nanmax(flat)
            pad = max(0.05, 0.08 * (y_max - y_min if y_max > y_min else 1.0))
            ax.set_ylim(y_min - pad, y_max + pad)

        # leyenda (afuera a la derecha por defecto)
        if legend_outside:
            fig.subplots_adjust(right=0.82, top=0.90, left=0.10, bottom=0.12)
            ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, title="Frecuencia")
        else:
            ax.legend(frameon=False)

        out_file = out_dir / f"{metric}_UE{ue}.png"
        fig.savefig(out_file, dpi=180, bbox_inches="tight")
        plt.close(fig)


def _axis_format_db(ax):
    sf = ScalarFormatter(useOffset=False)
    sf.set_scientific(False)
    ax.yaxis.set_major_formatter(sf)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)


def plot_all_metrics_combined(df_all: pd.DataFrame, out_dir: Path):
    """
    Una imagen por UE con 5 subplots (PRx, SINR, SE combinado, TBLER step, TBLER running).
    - Título incluye UE y 'freqs: ...'
    - SE (Shannon vs SE-LA) ocupa el doble de altura (GridSpec con ratios)
    - 'Step' en TODOS los subplots
    - Leyendas debajo de cada subplot (una por subplot, no global)
    """
    ue_ids = sorted(df_all["ue_id"].dropna().astype(int).unique().tolist())
    freqs = sorted(df_all["freq_mhz"].dropna().unique().tolist())
    label_for = {f: f"{f:.0f} MHz" for f in freqs}
    freqs_str = ", ".join([label_for[f] for f in freqs])

    for ue in ue_ids:
        # --- Más espacio vertical para las leyendas
        fig = plt.figure(figsize=(13, 12))
        gs = GridSpec(nrows=4, ncols=2, height_ratios=[1.1, 2.4, 1.3, 0.3], hspace=1.4, wspace=0.32)

        ax_prx   = fig.add_subplot(gs[0, 0])  # PRx
        ax_sinr  = fig.add_subplot(gs[0, 1])  # SINR
        ax_se    = fig.add_subplot(gs[1, :])  # SE combinado (doble altura)
        ax_tbl_s = fig.add_subplot(gs[2, 0])  # TBLER step
        ax_tbl_r = fig.add_subplot(gs[2, 1])  # TBLER running
        ax_dummy = fig.add_subplot(gs[3, :])  # espacio libre
        ax_dummy.axis("off")

        y_collect = {k: [] for k in ["prx", "sinr", "se", "tbl_s", "tbl_r"]}

        for j, f in enumerate(freqs):
            df_f = df_all[(df_all["ue_id"] == ue) & (df_all["freq_mhz"] == f)].sort_values("step")
            x = df_f["step"].to_numpy(dtype=float)
            x_off = x + _X_OFFSETS[j % len(_X_OFFSETS)]

            # PRx (medido)
            y = df_f["prx_dbm"].to_numpy(dtype=float)
            ax_prx.plot(x_off, y, marker=_MARKERS[j % len(_MARKERS)],
                        linestyle=_LINESTY[j % len(_LINESTY)], linewidth=1.6,
                        label=label_for[f])
            y_collect["prx"].append(y[~np.isnan(y)])

            # PRx teórico
            y_th = df_f["prx_dbm_theo"].to_numpy(dtype=float)
            ax_prx.plot(x_off, y_th, linestyle=":", linewidth=2.0,
                        label=f"{label_for[f]} · Teórico")
            y_collect["prx"].append(y_th[~np.isnan(y_th)])

            # SINR
            y = df_f["sinr_eff_db"].to_numpy(dtype=float)
            ax_sinr.plot(x_off, y, marker=_MARKERS[j % len(_MARKERS)],
                         linestyle=_LINESTY[j % len(_LINESTY)], linewidth=1.6,
                         label=label_for[f])
            y_collect["sinr"].append(y[~np.isnan(y)])

            # SE combinado
            y_sh = df_f["se_shannon"].to_numpy(dtype=float)
            y_la = df_f["se_la"].to_numpy(dtype=float)
            ax_se.plot(x_off, y_sh, marker=_MARKERS[j % len(_MARKERS)],
                       linestyle="-", linewidth=1.8, label=f"{label_for[f]} · Shannon")
            ax_se.plot(x_off, y_la, marker=_MARKERS[j % len(_MARKERS)],
                       linestyle="--", linewidth=1.6, label=f"{label_for[f]} · SE-LA")
            y_collect["se"].append(y_sh[~np.isnan(y_sh)])
            y_collect["se"].append(y_la[~np.isnan(y_la)])

            # TBLER step
            y = df_f["tbler_step"].to_numpy(dtype=float)
            ax_tbl_s.plot(x_off, y, marker=_MARKERS[j % len(_MARKERS)],
                          linestyle=_LINESTY[j % len(_LINESTY)], linewidth=1.6,
                          label=label_for[f])
            y_collect["tbl_s"].append(y[~np.isnan(y)])

            # TBLER running
            y = df_f["tbler_running"].to_numpy(dtype=float)
            ax_tbl_r.plot(x_off, y, marker=_MARKERS[j % len(_MARKERS)],
                          linestyle=_LINESTY[j % len(_LINESTY)], linewidth=1.6,
                          label=label_for[f])
            y_collect["tbl_r"].append(y[~np.isnan(y)])

        # === Títulos y etiquetas ===
        ax_prx.set_title("PRx (dBm)")
        ax_sinr.set_title("SINR (dB)")
        ax_se.set_title("SE — Shannon (cota) vs SE-LA (real)")
        ax_tbl_s.set_title("TBLER (por step)")
        ax_tbl_r.set_title("TBLER running")

        for ax, ylabel in zip(
            [ax_prx, ax_sinr, ax_se, ax_tbl_s, ax_tbl_r],
            ["PRx (dBm)", "SINR (dB)", "SE (b/s/Hz)", "TBLER", "TBLER"]
        ):
            ax.set_ylabel(ylabel)
            ax.set_xlabel("Step")
            ax.grid(True, linestyle="--", alpha=0.35)
            ax.set_axisbelow(True)

        _axis_format_db(ax_prx)
        _axis_format_db(ax_sinr)

        # === Padding dinámico ===
        for key, ax in [("prx", ax_prx), ("sinr", ax_sinr), ("se", ax_se), ("tbl_s", ax_tbl_s), ("tbl_r", ax_tbl_r)]:
            flat = np.concatenate([v for v in y_collect[key] if v.size > 0]) if any(len(v) > 0 for v in y_collect[key]) else np.array([])
            if flat.size > 0:
                y_min, y_max = np.nanmin(flat), np.nanmax(flat)
                pad = max(0.05, 0.08 * (y_max - y_min if y_max > y_min else 1.0))
                ax.set_ylim(y_min - pad, y_max + pad)

        # === Leyendas debajo de cada subplot ===
        def add_legend_below(ax, dy=-0.38):
            h, l = ax.get_legend_handles_labels()
            if h:
                ax.legend(
                    h, l,
                    loc="upper center",
                    bbox_to_anchor=(0.5, dy),
                    frameon=False,
                    ncol=min(3, max(1, len(l) // 2)),
                    fontsize=9
                )

        # Las normales
        for ax in [ax_prx, ax_sinr, ax_tbl_s, ax_tbl_r]:
            add_legend_below(ax, dy=-0.38)

        # La SE (más alta)
        add_legend_below(ax_se, dy=-0.25)

        # Ajuste global
        fig.subplots_adjust(top=0.94, bottom=0.08, left=0.07, right=0.97, hspace=1.6)
        fig.suptitle(f"UE {ue} — freqs: {freqs_str}", fontsize=14, y=0.995)

        out_file = out_dir / f"UE{ue}_all_metrics.png"
        fig.savefig(out_file, dpi=180, bbox_inches="tight")
        plt.close(fig)


def plot_all_metrics_single_freq(df_all: pd.DataFrame, freq_mhz: float, out_dir: Path):
    """
    Una imagen por UE mostrando SOLO la frecuencia 'freq_mhz'.
    PRx incluye la cota teórica (prx_dbm_theo) si está disponible.
    Layout: SE con doble altura y 'Step' en todos los subplots.
    """
    df_f_all = df_all[np.isclose(df_all["freq_mhz"], freq_mhz)].copy()
    if df_f_all.empty:
        print(f"[WARN] No hay datos para {freq_mhz} MHz")
        return

    ue_ids = sorted(df_f_all["ue_id"].dropna().astype(int).unique().tolist())
    label = f"{freq_mhz:.0f} MHz"
    has_prx_theo = "prx_dbm_theo" in df_f_all.columns

    for ue in ue_ids:
        # Figura con más separación vertical (sin tight_layout)
        fig = plt.figure(figsize=(13, 11))
        gs = GridSpec(nrows=4, ncols=2, height_ratios=[1.1, 2.4, 1.3, 0.3], hspace=1.3, wspace=0.32)
        ax_prx   = fig.add_subplot(gs[0, 0])
        ax_sinr  = fig.add_subplot(gs[0, 1])
        ax_se    = fig.add_subplot(gs[1, :])
        ax_tbl_s = fig.add_subplot(gs[2, 0])
        ax_tbl_r = fig.add_subplot(gs[2, 1])
        ax_dummy = fig.add_subplot(gs[3, :]); ax_dummy.axis("off")

        df_f = df_f_all[(df_f_all["ue_id"] == ue)].sort_values("step")
        x = df_f["step"].to_numpy(dtype=float)

        # ========= PRx =========
        y_prx = df_f["prx_dbm"].to_numpy(dtype=float)
        line_meas, = ax_prx.plot(x, y_prx, marker="o", linestyle="-", linewidth=1.8, label=label)

        if has_prx_theo:
            y_th = df_f["prx_dbm_theo"].to_numpy(dtype=float)
            ax_prx.plot(x, y_th, linestyle=":", linewidth=2.0,
                        color=line_meas.get_color(), label=f"{label} · Teórico")

        # ========= SINR =========
        ax_sinr.plot(x, df_f["sinr_eff_db"], marker="o", linestyle="-", linewidth=1.8, label=label)

        # ========= SE =========
        y_sh = df_f["se_shannon"].to_numpy(dtype=float)
        y_la = df_f["se_la"].to_numpy(dtype=float)
        ax_se.plot(x, y_sh, marker="o", linestyle="-",  linewidth=1.9, label=f"{label} · Shannon")
        ax_se.plot(x, y_la, marker="s", linestyle="--", linewidth=1.7, label=f"{label} · SE-LA")

        # ========= TBLER step =========
        ax_tbl_s.plot(x, df_f["tbler_step"], marker="o", linestyle="-", linewidth=1.8, label=label)

        # ========= TBLER running =========
        ax_tbl_r.plot(x, df_f["tbler_running"], marker="o", linestyle="-", linewidth=1.8, label=label)

        # === Configuración de ejes ===
        ax_prx.set_title("PRx (dBm)");            ax_prx.set_ylabel("PRx (dBm)")
        ax_sinr.set_title("SINR (dB)");           ax_sinr.set_ylabel("SINR (dB)")
        ax_se.set_title("SE — Shannon (cota) vs SE-LA (real)"); ax_se.set_ylabel("SE (b/s/Hz)")
        ax_tbl_s.set_title("TBLER (por step)");   ax_tbl_s.set_ylabel("TBLER")
        ax_tbl_r.set_title("TBLER running");      ax_tbl_r.set_ylabel("TBLER")

        for ax in [ax_prx, ax_sinr, ax_se, ax_tbl_s, ax_tbl_r]:
            ax.set_xlabel("Step")
            ax.grid(True, linestyle="--", alpha=0.35)
            ax.set_axisbelow(True)

        _axis_format_db(ax_prx)
        _axis_format_db(ax_sinr)

        # === Límites con padding ===
        def _pad(ax, *arrays):
            vals = []
            for a in arrays:
                if a is not None:
                    a = a[~np.isnan(a)]
                    if a.size: vals.append(a)
            if vals:
                yv = np.concatenate(vals)
                y_min, y_max = np.min(yv), np.max(yv)
                pad = max(0.05, 0.08 * (y_max - y_min if y_max > y_min else 1.0))
                ax.set_ylim(y_min - pad, y_max + pad)

        _pad(ax_prx,  y_prx, (df_f["prx_dbm_theo"].to_numpy(dtype=float) if has_prx_theo else None))
        _pad(ax_sinr, df_f["sinr_eff_db"].to_numpy(dtype=float))
        _pad(ax_se,   y_sh, y_la)
        _pad(ax_tbl_s, df_f["tbler_step"].to_numpy(dtype=float))
        _pad(ax_tbl_r, df_f["tbler_running"].to_numpy(dtype=float))

        # === Leyendas debajo de cada subplot ===
        def add_legend_below(ax, dy=-0.42):
            h, l = ax.get_legend_handles_labels()
            if h:
                ax.legend(
                    h, l,
                    loc="upper center",
                    bbox_to_anchor=(0.5, dy),
                    frameon=False,
                    ncol=2,
                    fontsize=9
                )

        # Ajuste más fino de las leyendas (SE más pegada)
        for ax in [ax_prx, ax_sinr, ax_tbl_s, ax_tbl_r]:
            add_legend_below(ax, dy=-0.42)

        # La de SE va más arriba porque la gráfica es más alta
        add_legend_below(ax_se, dy=-0.25)

        # === Ajuste global de márgenes ===
        fig.subplots_adjust(top=0.94, bottom=0.08, left=0.07, right=0.97, hspace=1.6)

        fig.suptitle(f"UE {ue} — freq: {label}", fontsize=14, y=0.995)

        out_file = out_dir / f"UE{ue}_all_metrics_{int(freq_mhz)}MHz.png"
        fig.savefig(out_file, dpi=180, bbox_inches="tight")
        plt.close(fig)

def _ue_labels(df_freq):
    ue_ids = sorted(df_freq["ue_id"].dropna().astype(int).unique().tolist())
    return ue_ids, [f"UE{i}" for i in ue_ids]

#Crea la gráfica de potencia recibida con los subplot para cada receptor
def plot_all_ues_prx_by_freq(df_all: pd.DataFrame, freq_mhz: float, out_dir: Path,
                             show_theoretical: bool = True):
    """
    Muestra la potencia recibida (PRx) por step para todos los UEs en una figura con subplots.
    No reescala los valores, muestra opcionalmente la potencia teórica,
    y mantiene ejes X e Y consistentes visualmente.
    """
    df_f = df_all[np.isclose(df_all["freq_mhz"], freq_mhz)].copy()
    if df_f.empty:
        print(f"[WARN] No hay datos para {freq_mhz} MHz")
        return

    ue_ids, _ = _ue_labels(df_f)
    n_ues = len(ue_ids)
    ncols = 3
    nrows = int(np.ceil(n_ues / ncols))

    # Determinar rango global Y
    y_min = df_f["prx_dbm"].min()
    y_max = df_f["prx_dbm"].max()
    if show_theoretical and "prx_dbm_theo" in df_f.columns:
        y_min = min(y_min, df_f["prx_dbm_theo"].min())
        y_max = max(y_max, df_f["prx_dbm_theo"].max())
    margin_y = 0.05 * (y_max - y_min)
    y_lim = (y_min - margin_y, y_max + margin_y)

    # Determinar rango global X
    x_min = df_f["step"].min()
    x_max = df_f["step"].max()
    margin_x = 0.02 * (x_max - x_min)
    x_lim = (x_min - margin_x, x_max + margin_x)

    # Ajuste automático del tamaño horizontal
    fig_width = max(13, ncols * 4.3)
    fig_height = max(7, nrows * 3.0)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                             figsize=(fig_width, fig_height),
                             sharex=True, sharey=True)
    axes = axes.flatten()

    for i, ue in enumerate(ue_ids):
        ax = axes[i]
        dfx = df_f[df_f["ue_id"] == ue].sort_values("step")

        x = dfx["step"].to_numpy(float)
        y = dfx["prx_dbm"].to_numpy(float)
        ax.plot(x, y, marker="o", linestyle="-", linewidth=1.8, label="PRx simulado")

        if show_theoretical and "prx_dbm_theo" in dfx.columns:
            y_th = dfx["prx_dbm_theo"].to_numpy(float)
            ax.plot(x, y_th, linestyle=":", linewidth=2.0, label="PRx teórico")

        ax.set_title(f"UE {ue}")
        ax.set_xlabel("Step")
        ax.set_ylabel("PRx [dBm]")
        ax.set_ylim(y_lim)
        ax.set_xlim(x_lim)
        ax.grid(True, linestyle="--", alpha=0.4)

        # Forzar ticks visibles en X e Y
        ax.tick_params(axis="x", which="both", labelbottom=True)
        ax.tick_params(axis="y", which="both", labelleft=True)

    # Eliminar subplots vacíos
    for j in range(len(ue_ids), len(axes)):
        fig.delaxes(axes[j])

    # Título general
    pt_dbm = df_f["ptx_dbm"].iloc[0] if "ptx_dbm" in df_f.columns else None
    title = f"Potencia recibida (PRx) [dBm] — Frecuencia: {freq_mhz:.0f} MHz"
    if pt_dbm is not None:
        title += f" · Potencia: {pt_dbm:.1f} dBm"
    fig.suptitle(title, fontsize=13, y=0.98)

    # Leyenda global
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out = out_dir / f"PRx's_dBm_{int(freq_mhz)}MHz.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)

#Crea la gráfica de SINR con subplots para cada receptor
def plot_all_ues_sinr_by_freq(df_all: pd.DataFrame, freq_mhz: float, out_dir: Path):
    """
    Muestra el SINR [dB] por step para todos los UEs en una figura con subplots.
    No reescala los valores individualmente y mantiene ejes X e Y consistentes visualmente.
    """
    # Filtrar frecuencia
    df_f = df_all[np.isclose(df_all["freq_mhz"], freq_mhz)].copy()
    if df_f.empty:
        print(f"[WARN] No hay datos para {freq_mhz} MHz")
        return

    # Identificar los UEs presentes
    ue_ids, _ = _ue_labels(df_f)
    n_ues = len(ue_ids)
    ncols = 3
    nrows = int(np.ceil(n_ues / ncols))

    # Determinar rango global Y (SINR)
    y_min = df_f["sinr_eff_db"].min()
    y_max = df_f["sinr_eff_db"].max()
    margin_y = 0.05 * (y_max - y_min)
    y_lim = (y_min - margin_y, y_max + margin_y)

    # Determinar rango global X (Steps)
    x_min = df_f["step"].min()
    x_max = df_f["step"].max()
    margin_x = 0.02 * (x_max - x_min)
    x_lim = (x_min - margin_x, x_max + margin_x)

    # Ajuste automático del tamaño de figura
    fig_width = max(13, ncols * 4.3)
    fig_height = max(7, nrows * 3.0)

    # Crear subplots
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                             figsize=(fig_width, fig_height),
                             sharex=True, sharey=True)
    axes = axes.flatten()

    # Graficar cada UE
    for i, ue in enumerate(ue_ids):
        ax = axes[i]
        dfx = df_f[df_f["ue_id"] == ue].sort_values("step")

        x = dfx["step"].to_numpy(float)
        y = dfx["sinr_eff_db"].to_numpy(float)
        ax.plot(x, y, marker="o", linestyle="-", linewidth=1.8, label="SINR")

        ax.set_title(f"UE {ue}")
        ax.set_xlabel("Step")
        ax.set_ylabel("SINR [dB]")
        ax.set_ylim(y_lim)
        ax.set_xlim(x_lim)
        ax.grid(True, linestyle="--", alpha=0.4)

        # Forzar ticks visibles
        ax.tick_params(axis="x", which="both", labelbottom=True)
        ax.tick_params(axis="y", which="both", labelleft=True)

    # Eliminar subplots vacíos
    for j in range(len(ue_ids), len(axes)):
        fig.delaxes(axes[j])

    # Título general
    pt_dbm = df_f["ptx_dbm"].iloc[0] if "ptx_dbm" in df_f.columns else None
    title = f"SINR [dB] — Frecuencia: {freq_mhz:.0f} MHz"
    if pt_dbm is not None:
        title += f" · Potencia: {pt_dbm:.1f} dBm"
    fig.suptitle(title, fontsize=13, y=0.98)

    # Leyenda global
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right")

    # Ajuste final
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out = out_dir / f"SINR's_db_{int(freq_mhz)}MHz.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)

#Crea la gráfica de SE_LA vs Shannon con subplots para cada receptor
def plot_all_ues_se_comparison(df_all: pd.DataFrame, freq_mhz: float, out_dir: Path):
    """
    Muestra SE_LA y SE_Shannon por step para todos los UEs en subplots.
    SE_LA se grafica en azul y SE_Shannon en naranja.
    """
    # Filtrar frecuencia
    df_f = df_all[np.isclose(df_all["freq_mhz"], freq_mhz)].copy()
    if df_f.empty:
        print(f"[WARN] No hay datos para {freq_mhz} MHz")
        return

    # Identificar los UEs presentes
    ue_ids, _ = _ue_labels(df_f)
    n_ues = len(ue_ids)
    ncols = 3
    nrows = int(np.ceil(n_ues / ncols))

    # Rango global de Y (para ambos SE)
    y_min = df_f[["se_la", "se_shannon"]].min().min()
    y_max = df_f[["se_la", "se_shannon"]].max().max()
    margin_y = 0.05 * (y_max - y_min)
    y_lim = (y_min - margin_y, y_max + margin_y)

    # Rango global de X (steps)
    x_min = df_f["step"].min()
    x_max = df_f["step"].max()
    margin_x = 0.02 * (x_max - x_min)
    x_lim = (x_min - margin_x, x_max + margin_x)

    # Ajuste del tamaño de figura
    fig_width = max(13, ncols * 4.3)
    fig_height = max(7, nrows * 3.0)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                             figsize=(fig_width, fig_height),
                             sharex=True, sharey=True)
    axes = axes.flatten()

    for i, ue in enumerate(ue_ids):
        ax = axes[i]
        dfx = df_f[df_f["ue_id"] == ue].sort_values("step")

        x = dfx["step"].to_numpy(float)
        y_la = dfx["se_la"].to_numpy(float)
        y_sh = dfx["se_shannon"].to_numpy(float)

        ax.plot(x, y_la, marker="o", linestyle="-", linewidth=1.8, color="tab:blue", label="SE_LA")
        ax.plot(x, y_sh, marker="s", linestyle="--", linewidth=1.8, color="tab:orange", label="SE_Shannon")

        ax.set_title(f"UE {ue}")
        ax.set_xlabel("Step")
        ax.set_ylabel("SE [bps/Hz]")
        ax.set_ylim(y_lim)
        ax.set_xlim(x_lim)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.tick_params(axis="x", which="both", labelbottom=True)
        ax.tick_params(axis="y", which="both", labelleft=True)

    # Eliminar subplots vacíos
    for j in range(len(ue_ids), len(axes)):
        fig.delaxes(axes[j])

    # Título general
    title = f"Eficiencia espectral (Teórica vs Real) — Frecuencia: {freq_mhz:.0f} MHz"
    fig.suptitle(title, fontsize=13, y=0.98)

    # Leyenda global
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out = out_dir / f"SE's_comparison_{int(freq_mhz)}MHz.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_all_ues_tbler_step_by_freq(
    df_all: pd.DataFrame,
    freq_mhz: float,
    out_dir: Path,
    ncols: int = 3,
    y_soft_bounds: tuple[float, float] = (-0.15, 1.15),  # rango amplio solicitado
    legend_outside: bool = True
):
    """
    SOLO TBLER STEP por UE en subplots.
    - Ejes compartidos.
    - Leyenda global.
    - Rango Y amplio por defecto (-0.5..1.5).
    """
    df_f = df_all[np.isclose(df_all["freq_mhz"], freq_mhz)].copy()
    if df_f.empty:
        print(f"[WARN] No hay datos para {freq_mhz} MHz")
        return

    # UEs y grilla
    ue_ids = sorted(df_f["ue_id"].dropna().astype(int).unique().tolist())
    n_ues = len(ue_ids)
    nrows = int(np.ceil(n_ues / ncols))

    # ----- Rango X (steps) -----
    x_min = float(df_f["step"].min())
    x_max = float(df_f["step"].max())
    margin_x = 0.02 * max(x_max - x_min, 1.0)
    x_lim = (x_min - margin_x, x_max + margin_x)

    # ----- Rango Y (amplio) -----
    y_lim = y_soft_bounds

    # ----- Figura -----
    fig_width = max(13, ncols * 4.3)
    fig_height = max(7, nrows * 3.0)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                             figsize=(fig_width, fig_height),
                             sharex=True, sharey=True)
    axes = axes.flatten()

    main_label = "TBLER (step)"
    plotted_any = False
    saved_handles = saved_labels = None

    for i, ue in enumerate(ue_ids):
        ax = axes[i]
        dfx = df_f[df_f["ue_id"] == ue].sort_values("step")
        x = dfx["step"].to_numpy(float)
        y = dfx["tbler_step"].astype(float).to_numpy() if "tbler_step" in dfx.columns else np.full_like(x, np.nan)

        if np.all(np.isnan(y)):
            ax.text(0.5, 0.5, "Sin datos", transform=ax.transAxes,
                    ha="center", va="center", fontsize=10, alpha=0.7)
        else:
            h = ax.plot(x, y, marker="o", linestyle="-", linewidth=1.8, label=main_label)
            if not plotted_any:
                saved_handles, saved_labels = ax.get_legend_handles_labels()
            plotted_any = True

        ax.set_title(f"UE {ue}")
        ax.set_xlabel("Step")
        ax.set_ylabel("TBLER")
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.tick_params(axis="x", which="both", labelbottom=True)
        ax.tick_params(axis="y", which="both", labelleft=True)

    # Subplots sobrantes
    for j in range(len(ue_ids), len(axes)):
        fig.delaxes(axes[j])

    # Título global
    pt_dbm = df_f["ptx_dbm"].iloc[0] if "ptx_dbm" in df_f.columns else None
    title = f"TBLER (step) — Frecuencia: {freq_mhz:.0f} MHz"
    if pt_dbm is not None:
        title += f" · Potencia: {pt_dbm:.1f} dBm"
    fig.suptitle(title, fontsize=13, y=0.98)

    # Leyenda global
    if plotted_any and saved_handles:
        loc = "upper right" if legend_outside else "best"
        fig.legend(saved_handles, saved_labels, loc=loc)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out = out_dir / f"TBLER's_step_{int(freq_mhz)}MHz.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_all_ues_tbler_running_by_freq(
    df_all: pd.DataFrame,
    freq_mhz: float,
    out_dir: Path,
    ncols: int = 3,
    y_soft_bounds: tuple[float, float] = (-0.15, 1.15),  # rango amplio solicitado
    legend_outside: bool = True
):
    """
    NUEVA: SOLO TBLER RUNNING por UE en subplots.
    - Ejes compartidos.
    - Leyenda global.
    - Rango Y amplio por defecto (-0.5..1.5).
    """
    df_f = df_all[np.isclose(df_all["freq_mhz"], freq_mhz)].copy()
    if df_f.empty:
        print(f"[WARN] No hay datos para {freq_mhz} MHz")
        return
    if "tbler_running" not in df_f.columns:
        print("[WARN] DataFrame no contiene 'tbler_running'")
        return

    # UEs y grilla
    ue_ids = sorted(df_f["ue_id"].dropna().astype(int).unique().tolist())
    n_ues = len(ue_ids)
    nrows = int(np.ceil(n_ues / ncols))

    # ----- Rango X (steps) -----
    x_min = float(df_f["step"].min())
    x_max = float(df_f["step"].max())
    margin_x = 0.02 * max(x_max - x_min, 1.0)
    x_lim = (x_min - margin_x, x_max + margin_x)

    # ----- Rango Y (amplio) -----
    y_lim = y_soft_bounds

    # ----- Figura -----
    fig_width = max(13, ncols * 4.3)
    fig_height = max(7, nrows * 3.0)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                             figsize=(fig_width, fig_height),
                             sharex=True, sharey=True)
    axes = axes.flatten()

    main_label = "TBLER (running)"
    plotted_any = False
    saved_handles = saved_labels = None

    for i, ue in enumerate(ue_ids):
        ax = axes[i]
        dfx = df_f[df_f["ue_id"] == ue].sort_values("step")
        x = dfx["step"].to_numpy(float)
        y = dfx["tbler_running"].astype(float).to_numpy()

        if np.all(np.isnan(y)):
            ax.text(0.5, 0.5, "Sin datos", transform=ax.transAxes,
                    ha="center", va="center", fontsize=10, alpha=0.7)
        else:
            ax.plot(x, y, linestyle="-", linewidth=1.8, label=main_label)
            if not plotted_any:
                saved_handles, saved_labels = ax.get_legend_handles_labels()
            plotted_any = True

        ax.set_title(f"UE {ue}")
        ax.set_xlabel("Step")
        ax.set_ylabel("TBLER")
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.tick_params(axis="x", which="both", labelbottom=True)
        ax.tick_params(axis="y", which="both", labelleft=True)

    # Subplots sobrantes
    for j in range(len(ue_ids), len(axes)):
        fig.delaxes(axes[j])

    # Título global
    pt_dbm = df_f["ptx_dbm"].iloc[0] if "ptx_dbm" in df_f.columns else None
    title = f"TBLER (running) — Frecuencia: {freq_mhz:.0f} MHz"
    if pt_dbm is not None:
        title += f" · Potencia: {pt_dbm:.1f} dBm"
    fig.suptitle(title, fontsize=13, y=0.98)

    # Leyenda global
    if plotted_any and saved_handles:
        loc = "upper right" if legend_outside else "best"
        fig.legend(saved_handles, saved_labels, loc=loc)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out = out_dir / f"TBLER's_running_{int(freq_mhz)}MHz.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)

def plot_all_ues_metrics_by_freq(df_all: pd.DataFrame, freq_mhz: float, out_dir: Path):
    """
    Genera una figura por métrica (PRx, SINR, SE, TBLER-step, TBLER-running),
    donde cada figura muestra las curvas de todos los UEs para la frecuencia indicada.
    Usa desplazamientos y estilos distintos por UE para evitar solapamientos.
    """
    df_f = df_all[np.isclose(df_all["freq_mhz"], freq_mhz)].copy()
    if df_f.empty:
        print(f"[WARN] No hay datos para {freq_mhz} MHz")
        return

    ue_ids = sorted(df_f["ue_id"].unique())    

    metrics = {
        "prx_dbm": {
            "ylabel": "PRx [dBm]",
            "title": "Potencia recibida (PRx)",
            "filename": f"PRx_all_UEs_{int(freq_mhz)}MHz.png"
        },
        "sinr_eff_db": {
            "ylabel": "SINR [dB]",
            "title": "Relación señal/ruido (SINR)",
            "filename": f"SINR_all_UEs_{int(freq_mhz)}MHz.png"
        },
        "se_la": {
            "ylabel": "SE-LA [b/s/Hz]",
            "title": "Eficiencia espectral (SE-LA)",
            "filename": f"SE_LA_all_UEs_{int(freq_mhz)}MHz.png"
        },
        "tbler_step": {
            "ylabel": "TBLER",
            "title": "TBLER por step",
            "filename": f"TBLER_step_all_UEs_{int(freq_mhz)}MHz.png"
        },
        "tbler_running": {
            "ylabel": "TBLER",
            "title": "TBLER running",
            "filename": f"TBLER_running_all_UEs_{int(freq_mhz)}MHz.png"
        }
    }

    for metric, cfg in metrics.items():
        if metric not in df_f.columns:
            print(f"[WARN] No existe la métrica '{metric}' en el dataframe.")
            continue

        fig, ax = plt.subplots(figsize=(10, 5))

        for j, ue in enumerate(ue_ids):
            dfx = df_f[df_f["ue_id"] == ue].sort_values("step")

            # ✅ Desplazamiento y estilos distintos por UE
            x_offset = _X_OFFSETS[j % len(_X_OFFSETS)]
            x = dfx["step"].to_numpy(dtype=float) + x_offset
            y = dfx[metric].to_numpy(dtype=float)

            ax.plot(
                x, y,
                marker=_MARKERS[j % len(_MARKERS)],
                linestyle=_LINESTY[j % len(_LINESTY)],
                linewidth=1.8,
                markersize=5,
                label=f"UE {ue}"
            )

        ax.set_xlabel("Step")
        ax.set_ylabel(cfg["ylabel"])
        ax.set_title(f"{cfg['title']} — {freq_mhz:.0f} MHz", pad=10)
        ax.grid(True, linestyle="--", alpha=0.4)

        # ✅ Leyenda fuera del gráfico, común a todas las métricas
        ax.legend(
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            borderaxespad=0,
            fontsize=9,
            title="Usuarios (UE)"
        )

        # Ajuste de layout para dejar espacio a la leyenda
        fig.tight_layout(rect=[0, 0, 0.85, 1])

        out_path = out_dir / cfg["filename"]
        fig.savefig(out_path, dpi=180, bbox_inches="tight")
        plt.close(fig)



# ------ doppler ------
def _ue_labels_(df_f: pd.DataFrame):
    ue_ids = sorted(df_f["ue_id"].dropna().unique().tolist())
    labels = [f"UE {int(u)}" for u in ue_ids]
    return ue_ids, labels

def _no_sci_y(ax):
    """Desactiva notación científica en eje Y (y offset)"""
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=False))
    ax.ticklabel_format(style='plain', axis='y', useOffset=False)
    ax.get_yaxis().get_major_formatter().set_scientific(False)
    ax.get_yaxis().get_major_formatter().set_useOffset(False)

def _common_xy_limits(df_f: pd.DataFrame, ycol: str):
    x_min = df_f["step"].min()
    x_max = df_f["step"].max()
    margin_x = 0.02 * max(1, (x_max - x_min))
    x_lim = (x_min - margin_x, x_max + margin_x)

    y_vals = df_f[ycol].to_numpy(dtype=float)
    y_vals = y_vals[np.isfinite(y_vals)]
    if y_vals.size == 0:
        y_lim = (-1, 1)
    else:
        y_min, y_max = float(np.nanmin(y_vals)), float(np.nanmax(y_vals))
        if y_min == y_max:
            y_min -= 1.0; y_max += 1.0
        margin_y = 0.05 * (y_max - y_min)
        y_lim = (y_min - margin_y, y_max + margin_y)
    return x_lim, y_lim


def plot_fd_all_ues_onefig(df_all: pd.DataFrame, freq_mhz: float, out_dir: Path):
    df_f = df_all[np.isclose(df_all["freq_mhz"], freq_mhz)].copy()
    if df_f.empty or "doppler_fd_hz" not in df_f.columns:
        print(f"[WARN] No hay datos de doppler_fd_hz para {freq_mhz} MHz")
        return

    ue_ids, labels = _ue_labels_(df_f)
    x_lim, y_lim = _common_xy_limits(df_f, "doppler_fd_hz")

    fig, ax = plt.subplots(figsize=(12, 6))
    for ue, lbl in zip(ue_ids, labels):
        dfx = df_f[df_f["ue_id"] == ue].sort_values("step")
        x = dfx["step"].to_numpy(float)
        y = dfx["doppler_fd_hz"].to_numpy(float)
        ax.plot(x, y, marker="o", linewidth=1.8, label=lbl)

    ax.axhline(0.0, linestyle="--", linewidth=1.0, alpha=0.5)
    ax.set_title(f"Doppler estimado fD — {freq_mhz:.0f} MHz (todos los UEs)")
    ax.set_xlabel("Step")
    ax.set_ylabel("fD [Hz]")
    ax.set_xlim(x_lim); ax.set_ylim(y_lim)
    ax.grid(True, linestyle="--", alpha=0.4)
    _no_sci_y(ax)

    # leyenda compacta
    ax.legend(loc="upper left", ncol=2, fontsize=9)

    # explicación al pie
    expl = (
        "fD (Hz): frecuencia Doppler estimada por UE (a partir de la pendiente de fase entre símbolos). "
        "Su magnitud crece con velocidad y frecuencia portadora. Signo: acercamiento/alejamiento relativo. "
        "Impacto: a mayor |fD|, menor tiempo de coherencia; si no se compensa, puede degradar SINR/SE y aumentar TBLER."
    )
    fig.text(0.01, 0.01, expl, ha="left", va="bottom", fontsize=9, wrap=True)

    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    out = out_dir / f"Doppler_fD_ALL_{int(freq_mhz)}MHz.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_nu_tc_all_ues_onefig(df_all: pd.DataFrame, freq_mhz: float, out_dir: Path):
    df_f = df_all[np.isclose(df_all["freq_mhz"], freq_mhz)].copy()
    if df_f.empty or "doppler_nu" not in df_f.columns or "doppler_Tc_ms" not in df_f.columns:
        print(f"[WARN] No hay datos de doppler_nu/Tc para {freq_mhz} MHz")
        return

    ue_ids, labels = _ue_labels_(df_f)
    x_min = df_f["step"].min(); x_max = df_f["step"].max()
    margin_x = 0.02 * max(1, (x_max - x_min))
    x_lim = (x_min - margin_x, x_max + margin_x)

    # límites para nu
    y_vals = df_f["doppler_nu"].to_numpy(dtype=float)
    y_vals = y_vals[np.isfinite(y_vals)]
    if y_vals.size == 0:
        y_lim_nu = (0.0, 0.1)
    else:
        y_min, y_max = float(np.nanmin(y_vals)), float(np.nanmax(y_vals))
        if y_min == y_max:
            y_min -= 0.01; y_max += 0.01
        margin_y = 0.1 * (y_max - y_min)
        y_lim_nu = (max(0.0, y_min - margin_y), y_max + margin_y)

    fig, ax = plt.subplots(figsize=(12, 6))
    # ν por UE
    for ue, lbl in zip(ue_ids, labels):
        dfx = df_f[df_f["ue_id"] == ue].sort_values("step")
        x = dfx["step"].to_numpy(float)
        nu = dfx["doppler_nu"].to_numpy(float)
        ax.plot(x, nu, marker="o", linewidth=1.8, label=lbl)

    ax.set_title(f"Doppler normalizado ν=fD/SCS y tiempo de coherencia Tc — {freq_mhz:.0f} MHz (todos los UEs)")
    ax.set_xlabel("Step")
    ax.set_ylabel("ν (adimensional)")
    ax.set_xlim(x_lim); ax.set_ylim(y_lim_nu)
    ax.grid(True, linestyle="--", alpha=0.4)
    _no_sci_y(ax)

    # bandas guía de interpretación
    for lo, hi, name in [(0.0, 0.005, "despreciable"), (0.005, 0.02, "bajo"),
                         (0.02, 0.05, "moderado"), (0.05, 0.1, "alto")]:
        ax.axhspan(lo, hi, alpha=0.06)

    # Tc en eje derecho
    ax2 = ax.twinx()
    for ue in ue_ids:
        dfx = df_f[df_f["ue_id"] == ue].sort_values("step")
        x = dfx["step"].to_numpy(float)
        tc = dfx["doppler_Tc_ms"].to_numpy(float)
        ax2.plot(x, tc, linestyle=":", linewidth=1.5, alpha=0.7)
    ax2.set_ylabel("Tc [ms]")
    _no_sci_y(ax2)

    # leyenda (UEs) una vez
    ax.legend(loc="upper left", ncol=2, fontsize=9)

    expl = (
        "ν=fD/SCS: Doppler normalizado (cuánto representa el Doppler respecto al espaciado de subportadoras). "
        "≈0–0.005 despreciable, 0.005–0.02 bajo, 0.02–0.05 moderado, 0.05–0.1 alto, >0.1 severo (OFDM sufre sin compensación). "
        "Tc≈0.423/|fD|: tiempo de coherencia; si la duración del TB o ventana del step es >> Tc, el canal cambia dentro del bloque."
    )
    fig.text(0.01, 0.01, expl, ha="left", va="bottom", fontsize=9, wrap=True)

    plt.tight_layout(rect=[0, 0.06, 1, 0.96])
    out = out_dir / f"Doppler_nu_Tc_ALL_{int(freq_mhz)}MHz.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_slope_all_ues_onefig(df_all: pd.DataFrame, freq_mhz: float, out_dir: Path):
    df_f = df_all[np.isclose(df_all["freq_mhz"], freq_mhz)].copy()
    if df_f.empty or "doppler_slope_rad_per_sym" not in df_f.columns:
        print(f"[WARN] No hay datos de slope para {freq_mhz} MHz")
        return

    ue_ids, labels = _ue_labels_(df_f)
    x_lim, y_lim = _common_xy_limits(df_f, "doppler_slope_rad_per_sym")

    fig, ax = plt.subplots(figsize=(12, 6))
    for ue, lbl in zip(ue_ids, labels):
        dfx = df_f[df_f["ue_id"] == ue].sort_values("step")
        x = dfx["step"].to_numpy(float)
        y = dfx["doppler_slope_rad_per_sym"].to_numpy(float)
        ax.plot(x, y, marker="o", linewidth=1.8, label=lbl)

    ax.axhline(0.0, linestyle="--", linewidth=1.0, alpha=0.5)
    ax.set_title(f"Pendiente de fase por símbolo — {freq_mhz:.0f} MHz (todos los UEs)")
    ax.set_xlabel("Step")
    ax.set_ylabel("slope [rad/símb]")
    ax.set_xlim(x_lim); ax.set_ylim(y_lim)
    ax.grid(True, linestyle="--", alpha=0.4)
    _no_sci_y(ax)

    ax.legend(loc="upper left", ncol=2, fontsize=9)

    expl = (
        "slope (rad/símb): rotación media de la fase por símbolo OFDM. "
        "slope = 2π·fD·Tsym; crece con fD o con símbolos más largos (SCS menor). "
        "Si no se compensa, rotaciones grandes desalinean precoder/igualador y pueden degradar las métricas."
    )
    fig.text(0.01, 0.01, expl, ha="left", va="bottom", fontsize=9, wrap=True)

    plt.tight_layout(rect=[0, 0.06, 1, 0.96])
    out = out_dir / f"Doppler_slope_ALL_{int(freq_mhz)}MHz.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)



def main():
    start_time = time.perf_counter()

    print(f"[INFO] Guardando resultados en: {OUT_DIR}")

    
    # 2) Corre TODAS las frecuencias con el MISMO plan
    runs = []
    for f in FREQS_MHZ:
        print(f"[RUN] Episodio @ {f:.0f} MHz")
        r = run_episode(f)
        runs.append(r)

    # 3) ... tu pipeline de plots/CSV/mapas sigue igual
    df_list = [to_dataframe(r) for r in runs]
    df_all = pd.concat(df_list, ignore_index=True)
    (OUT_DIR / "compare_metrics_all.csv").write_text(df_all.to_csv(index=False))


    #Total de metricas por receptor y frecuencia combinadas
    plot_all_metrics_combined(df_all, OUT_DIR)

    #Metricas por usuario
    plot_metric_per_ue(df_all, metric="prx_dbm",       ylabel="PRx (dBm)",            out_dir=OUT_DIR_UE_METRICS)
    plot_metric_per_ue(df_all, metric="sinr_eff_db",   ylabel="SINR efectivo (dB)",   out_dir=OUT_DIR_UE_METRICS)
    plot_metric_per_ue(df_all, metric="se_la",         ylabel="SE (LA) [bits/s/Hz]",  out_dir=OUT_DIR_UE_METRICS)
    plot_metric_per_ue(df_all, metric="se_shannon",    ylabel="SE (Shannon) [b/s/Hz]",out_dir=OUT_DIR_UE_METRICS)
    plot_metric_per_ue(df_all, metric="tbler_step",    ylabel="TBLER (por step)",     out_dir=OUT_DIR_UE_METRICS)
    plot_metric_per_ue(df_all, metric="tbler_running", ylabel="TBLER running",        out_dir=OUT_DIR_UE_METRICS)


    
    # plots totales por frecuencia
    for f in FREQS_MHZ:
       plot_all_metrics_single_freq(df_all, f, OUT_DIR_UE_all_METRICS) #Metricas totales por usuario y frecuencia
       plot_all_ues_metrics_by_freq(df_all, f, OUT_DIR_FREQ_METRICS) #Metricas totales por frecuencia

    #Receptors metrics
    for f in FREQS_MHZ:
        plot_all_ues_prx_by_freq(df_all, f, OUT_DIR_RECEPTORS)                #Solo PRx
        plot_all_ues_se_comparison(df_all, f, OUT_DIR_RECEPTORS)
        plot_all_ues_sinr_by_freq(df_all, f, OUT_DIR_RECEPTORS)               #Solo SINR
        plot_all_ues_tbler_step_by_freq(df_all, f, OUT_DIR_RECEPTORS)         #Solo TBLER step
        plot_all_ues_tbler_running_by_freq(df_all, f, OUT_DIR_RECEPTORS)      #Solo TBLER running

    # === VISUALIZACIÓN FÍSICA ===
    print("[INFO] Generando gráficos de trayectoria y GIFs...")

    for r in runs:
        fmhz = r["freq_mhz"]
        tracks = r["tracks"]
        obstacles = r["obstacles"]
        bounds = r["bounds"]

        # 1. Plot Estático
        out_traj = OUT_DIR / f"traj_static_{SCENE}_{NUM_AGENTS} agentes_{SEMILLA} (seed)_{MAX_STEPS} steps.png"
        plot_trajectories_xy_xz(tracks, obstacles, bounds, out_path=out_traj,
                                freq_mhz=fmhz, num_agents=NUM_AGENTS, seed=SEMILLA)
        print(f"Imagen guardada: {out_traj}")

        # 2. GIF
        out_gif = OUT_DIR / f"animacion_{SCENE}_{NUM_AGENTS} agentes_{SEMILLA} (seed)_{MAX_STEPS} steps.gif"
        print("Generando GIF...")
        make_gif(tracks, obstacles, bounds, out_path=out_gif)
        print(f"GIF guardado: {out_gif}")
        
    #Doppler
    for f in FREQS_MHZ:
        plot_fd_all_ues_onefig(df_all, f, OUT_DIR_DOPPLER)
        plot_nu_tc_all_ues_onefig(df_all, f, OUT_DIR_DOPPLER)
        plot_slope_all_ues_onefig(df_all, f, OUT_DIR_DOPPLER)


    print(f"[DONE] Imágenes en: {OUT_DIR}")

    elapsed = time.perf_counter() - start_time
    print(f"Tiempo total transcurrido: {elapsed:.3f} s ({elapsed / 60:.2f} min)")


if __name__ == "__main__":
    main()
