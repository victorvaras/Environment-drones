#!/usr/bin/env python3

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["MPLBACKEND"] = "Agg"

import sys
import time
from datetime import datetime
from pathlib import Path
from time import perf_counter

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from env.environment.gymnasium_env import DroneEnv


SCENE = "simple_street_canyon"
DRONE_START = (0.0, 0.0, 20.0)
SEMILLA = 0
DT = 0.1
NUM_AGENTS = 5
RX_POSITIONS = None
RX_GOALS = None

MAX_STEPS = 100
FREQS_MHZ = [3500.0]

RUN_TAG = datetime.now().strftime("%Y%m%d-%H%M%S")
scene_clean = Path(SCENE).stem
OUT_DIR = Path(__file__).resolve().parent / f"PRUEBA_FUNCIONAMIENTO_{RUN_TAG}_{scene_clean}_{NUM_AGENTS} agentes_{SEMILLA} (seed)_{MAX_STEPS} steps"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _scalar_float(x):
    try:
        return float(x)
    except Exception:
        pass
    try:
        a = np.asarray(x)
        if a.shape == ():
            return float(a)
    except Exception:
        pass
    if hasattr(x, "value"):
        try:
            return float(x.value)
        except Exception:
            pass
    try:
        return np.float64(x).item()
    except Exception as exc:
        raise TypeError(f"No pude convertir a float el objeto de tipo {type(x)}: {x!r}") from exc


def _vec3_to_np(p):
    if isinstance(p, (list, tuple, np.ndarray)) and len(p) == 3:
        return np.array([_scalar_float(p[0]), _scalar_float(p[1]), _scalar_float(p[2])], dtype=float)

    if hasattr(p, "x") and hasattr(p, "y") and hasattr(p, "z"):
        return np.array([_scalar_float(p.x), _scalar_float(p.y), _scalar_float(p.z)], dtype=float)

    try:
        lst = list(p)
        if len(lst) >= 3:
            return np.array([_scalar_float(lst[0]), _scalar_float(lst[1]), _scalar_float(lst[2])], dtype=float)
    except Exception:
        pass

    try:
        arr = np.asarray(p, dtype=float).reshape(-1)
        if arr.size >= 3:
            return arr[:3]
    except Exception:
        pass

    raise TypeError(f"No pude extraer un vec3 de tipo {type(p)} ({p!r}).")


def _get_drone_xyz(rt) -> np.ndarray:
    if hasattr(rt, "dron") and hasattr(rt.dron, "pos"):
        return _vec3_to_np(rt.dron.pos)
    if hasattr(rt, "tx") and hasattr(rt.tx, "position"):
        return _vec3_to_np(rt.tx.position)
    if hasattr(rt, "txs") and isinstance(rt.txs, (list, tuple)) and rt.txs:
        tx0 = rt.txs[0]
        if hasattr(tx0, "position"):
            return _vec3_to_np(tx0.position)
    raise AttributeError("No pude obtener la posición del dron/tx (faltan atributos esperados).")


def _get_rx_positions_xyz(rt) -> np.ndarray:
    if hasattr(rt, "rx_list") and isinstance(rt.rx_list, (list, tuple)) and rt.rx_list:
        pos = []
        for rx in rt.rx_list:
            if hasattr(rx, "position"):
                pos.append(_vec3_to_np(rx.position))
            else:
                raise AttributeError("Un RX en rx_list no tiene atributo 'position'.")
        return np.vstack(pos).astype(float)

    if hasattr(rt, "receptores") and hasattr(rt.receptores, "positions_xyz"):
        arr = np.asarray(rt.receptores.positions_xyz())
        out = []
        for i in range(arr.shape[0]):
            out.append(_vec3_to_np(arr[i]))
        return np.vstack(out).astype(float)

    raise AttributeError("No pude obtener posiciones de receptores (rx_list o receptores.positions_xyz).")


def make_gif(tracks, obstacles, scene_bounds, out_path, fps=20):
    ues = tracks["ues"]
    drone = tracks["drone"]

    t_ues, n_ues, _ = ues.shape
    t_drone = drone.shape[0]
    total_steps = min(t_ues, t_drone)

    ues = ues[:total_steps]
    drone = drone[:total_steps]

    fig, ax = plt.subplots(figsize=(12, 12))
    fig.subplots_adjust(right=0.80)

    ax.set_aspect("equal")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_title(
        f"Simulación Dinámica (Sionna + SocialForce)\n"
        f"Escenario: {SCENE} | {n_ues} Agentes | Semilla N° {SEMILLA} ({total_steps - 1} pasos)",
        pad=12,
        fontsize=14,
        weight="bold",
    )
    ax.grid(True, alpha=0.3)

    if obstacles:
        obs_stack = np.vstack(obstacles)
        n_points = len(obs_stack)
        if n_points > 0:
            marker_size = 10000.0 / n_points
            marker_size = max(0.1, min(marker_size, 2.0))
            ax.scatter(obs_stack[:, 0], obs_stack[:, 1], s=marker_size, c="black", marker=".", alpha=1.0)

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

    cmap = matplotlib.colormaps["tab10"]
    colors = [cmap(i % 10) for i in range(n_ues)]

    start_pos_ues = ues[0, :, :2]
    scats_ues = ax.scatter(start_pos_ues[:, 0], start_pos_ues[:, 1], s=100, c=colors, zorder=5, edgecolors="white")
    trails_ues = [ax.plot([], [], "-", lw=2, color=colors[i], alpha=0.6)[0] for i in range(n_ues)]

    start_pos_drone = drone[0, :2]
    drone_scat = ax.scatter([start_pos_drone[0]], [start_pos_drone[1]], s=150, marker="^", color="tab:blue", edgecolors="k", zorder=6)
    drone_trail = ax.plot([], [], "--", lw=2.5, color="tab:blue", alpha=0.9, zorder=5)[0]

    def update(frame):
        current_pos_ues = ues[frame, :, :2]
        scats_ues.set_offsets(current_pos_ues)

        start_idx = max(0, frame - 30)
        for i, trail in enumerate(trails_ues):
            trail.set_data(ues[start_idx:frame + 1, i, 0], ues[start_idx:frame + 1, i, 1])

        current_pos_drone = drone[frame, :2]
        drone_scat.set_offsets(current_pos_drone)
        drone_trail.set_data(drone[start_idx:frame + 1, 0], drone[start_idx:frame + 1, 1])
        return scats_ues, drone_scat, drone_trail, *trails_ues

    target_total_frames = 250
    step_skip = 1 if total_steps <= target_total_frames else max(1, total_steps // target_total_frames)
    print(f"[GIF] Generando animación con step_skip = {step_skip} (Total frames: {total_steps // step_skip})")

    ani = animation.FuncAnimation(fig, update, frames=range(0, total_steps, step_skip), interval=50, blit=True)
    ani.save(out_path, writer="pillow", fps=fps)
    plt.close(fig)


def to_dataframe(run_dict: dict) -> pd.DataFrame:
    freq = run_dict["freq_mhz"]
    steps_ue_metrics = run_dict["steps_ue_metrics"]

    rows = []
    for step_idx, ue_list in enumerate(steps_ue_metrics, start=1):
        for ue_local_idx, metric in enumerate(ue_list):
            rows.append(
                {
                    "freq_mhz": freq,
                    "step": step_idx,
                    "ue_id": metric.get("ue_id", ue_local_idx),
                    "prx_dbm": float(metric.get("prx_dbm", np.nan)),
                    "prx_dbm_theo": float(metric.get("prx_dbm_theo", np.nan)),
                }
            )

    return pd.DataFrame(rows)


def plot_all_ues_prx_by_freq(df_all: pd.DataFrame, freq_mhz: float, out_dir: Path, show_theoretical: bool = True):
    df_f = df_all[np.isclose(df_all["freq_mhz"], freq_mhz)].copy()
    if df_f.empty:
        print(f"[WARN] No hay datos para {freq_mhz} MHz")
        return

    ue_ids = sorted(df_f["ue_id"].dropna().astype(int).unique().tolist())
    n_ues = len(ue_ids)
    ncols = 3
    nrows = int(np.ceil(n_ues / ncols))

    y_min = df_f["prx_dbm"].min()
    y_max = df_f["prx_dbm"].max()
    if show_theoretical and "prx_dbm_theo" in df_f.columns:
        y_min = min(y_min, df_f["prx_dbm_theo"].min())
        y_max = max(y_max, df_f["prx_dbm_theo"].max())

    margin_y = 0.05 * (y_max - y_min if y_max > y_min else 1.0)
    y_lim = (y_min - margin_y, y_max + margin_y)

    x_min = df_f["step"].min()
    x_max = df_f["step"].max()
    margin_x = 0.02 * (x_max - x_min if x_max > x_min else 1.0)
    x_lim = (x_min - margin_x, x_max + margin_x)

    fig_width = max(13, ncols * 4.3)
    fig_height = max(7, nrows * 3.0)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_width, fig_height), sharex=True, sharey=True)
    axes = np.atleast_1d(axes).flatten()

    for idx, ue in enumerate(ue_ids):
        ax = axes[idx]
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
        ax.tick_params(axis="x", which="both", labelbottom=True)
        ax.tick_params(axis="y", which="both", labelleft=True)

    for j in range(len(ue_ids), len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle(f"Potencia recibida (PRx) [dBm] — Frecuencia: {freq_mhz:.0f} MHz", fontsize=13, y=0.98)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out = out_dir / f"PRx_dBm_{int(freq_mhz)}MHz.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run_episode(freq_mhz: float) -> dict:
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
        mode_set_vuelo=7,
        run_metrics=True,
        step_durations=DT,
    )

    scene_bounds = env.scene_bounds
    env.reset(seed=SEMILLA)
    done = trunc = False

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

    steps_ue_metrics = []
    drone_traj, ue_traj, steps = [], [], []

    t = 0
    t_loop0 = perf_counter()
    while not (done or trunc):
        drone_traj.append(_get_drone_xyz(env.rt).copy())
        ue_traj.append(_get_rx_positions_xyz(env.rt).copy())
        steps.append(t)

        action_movement = [100.0, 0.0, 0.0, 20.0]
        
        _, _, done, trunc, info = env.step(action_movement)

        ue_metrics_step = info.get("ue_metrics", [])
        steps_ue_metrics.append([dict(m) for m in ue_metrics_step] if ue_metrics_step else [])
        t += 1

    t_loop = perf_counter() - t_loop0
    print(f"Loop wall-clock (mientras ejecuta steps): {t_loop:.6f} s")

    drone_traj.append(_get_drone_xyz(env.rt).copy())
    ue_traj.append(_get_rx_positions_xyz(env.rt).copy())
    steps.append(t)

    env.close()

    return {
        "freq_mhz": freq_mhz,
        "steps_ue_metrics": steps_ue_metrics,
        "obstacles": obstacles_np,
        "bounds": scene_bounds,
        "tracks": {
            "drone": np.vstack(drone_traj),
            "ues": np.stack(ue_traj, axis=0),
            "steps": np.array(steps, dtype=int),
        },
    }


def main() -> int:
    start_time = time.perf_counter()
    print(f"[INFO] Guardando resultados en: {OUT_DIR}")

    runs = []
    for freq in FREQS_MHZ:
        print(f"[RUN] Episodio @ {freq:.0f} MHz con {MAX_STEPS} steps")
        runs.append(run_episode(freq))

    df_list = [to_dataframe(r) for r in runs]
    df_all = pd.concat(df_list, ignore_index=True) if df_list else pd.DataFrame()

    csv_out = OUT_DIR / "prueba_funcionamiento_metrics.csv"
    df_all.to_csv(csv_out, index=False)
    print(f"[OK] CSV guardado: {csv_out}")

    for freq in FREQS_MHZ:
        plot_all_ues_prx_by_freq(df_all, freq, OUT_DIR)
        print(f"[OK] Gráfico PRx generado para {freq:.0f} MHz")

    for run in runs:
        gif_out = OUT_DIR / (
            f"animacion_{scene_clean}_{NUM_AGENTS} agentes_{SEMILLA} (seed)_"
            f"{MAX_STEPS} steps_{int(run['freq_mhz'])}MHz.gif"
        )
        print(f"[GIF] Generando: {gif_out.name}")
        make_gif(run["tracks"], run["obstacles"], run["bounds"], out_path=gif_out)
        print(f"[OK] GIF guardado: {gif_out}")

    elapsed = time.perf_counter() - start_time
    print(f"[DONE] Tiempo total: {elapsed:.3f} s ({elapsed / 60:.2f} min)")
    print(f"[DONE] Salida final: {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())