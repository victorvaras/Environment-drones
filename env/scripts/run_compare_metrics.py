# run_compare_metrics.py
# -*- coding: utf-8 -*-

# === Limpieza de logs===
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  #0 = all, 1 = info, 2 = warnings, 3 = solo errors

# === Bootstrap sys.path a la raíz del proyecto (dos niveles arriba) ===
import sys
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
from matplotlib.ticker import ScalarFormatter, MaxNLocator
from matplotlib.gridspec import GridSpec

# Desplazamientos y estilos por frecuencia para evitar solapamiento
_X_OFFSETS = [-0.12, -0.04, 0.04, 0.12, 0.20, -0.20]  # se cicla si hay >6 freqs
_MARKERS   = ["o", "s", "D", "^", "v", "P"]
_LINESTY   = ["-", "--", "-.", ":", "-", "--"]



from env.environment.gymnasium_env import DroneEnv  # adapta si tu ruta difiere


# ========= Configuración =========
SCENE = "simple_street_canyon_with_cars"  # p.ej. "santiago.xml", "munich"
DRONE_START = (0.0, 0.0, 10.0)
RX_POSITIONS = [
    (-50.0, 0.0, 1.5),
    (20.0, -30.0, 1.5),
    #(10.0, 0.0, 1.5),
    #(-10.0, 0.0, 1.5),
    #(-1.0, 0.0, 1.5),
    #(0.0,   30.0, 1.5),
    #(20.0,  -30.0, 1.5),
    #(80.0,   40.0, 1.5),
    #(50.0,    0.0, 1.5),
    #(90, -55, 1.5),
]
MAX_STEPS = 50

# Compara dos frecuencias (en MHz). Cambia a lo que necesites.
FREQS_MHZ = [3500.0, 28000]
FREQ_LABELS = [f"{f:.0f} MHz" for f in FREQS_MHZ]

# Carpeta de salida con timestamp
RUN_TAG = datetime.now().strftime("%Y%m%d-%H%M%S")
OUT_DIR = Path(project_root) / "outputs" / f"compare_metrics_{RUN_TAG}"
OUT_DIR.mkdir(parents=True, exist_ok=True)

import numpy as np


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


def run_episode(freq_mhz: float) -> dict:

    env = DroneEnv(
        render_mode=None,
        scene_name=SCENE,
        max_steps=MAX_STEPS,
        drone_start=DRONE_START,
        rx_positions=RX_POSITIONS if RX_POSITIONS else None,
        antenna_mode="SECTOR3_3GPP",
        frequency_mhz=freq_mhz,
    )

    obs, info = env.reset(seed=0)
    done = trunc = False

    steps_ue_metrics, steps_tbler_running = [], []
    drone_traj, ue_traj, steps = [], [], []

   


    t = 0
    while not (done or trunc):
        # snapshot antes del step
        drone_traj.append(_get_drone_xyz(env.rt).copy())
        ue_traj.append(_get_rx_positions_xyz(env.rt).copy())
        steps.append(t)

        a = [0,0,0]
        b = [0,0,0]

        obs, rew, done, trunc, info = env.step(a, b)

        ue_metrics_step = info.get("ue_metrics", [])
        tbler_running   = info.get("tbler_running_per_ue", None)
        steps_ue_metrics.append([dict(m) for m in ue_metrics_step])
        steps_tbler_running.append(list(tbler_running) if tbler_running is not None else [np.nan]*len(ue_metrics_step))

        t += 1

    # snapshot final
    drone_traj.append(_get_drone_xyz(env.rt).copy())
    ue_traj.append(_get_rx_positions_xyz(env.rt).copy())
    steps.append(t)

    env.close()

    return {
        "freq_mhz": freq_mhz,
        "steps_ue_metrics": steps_ue_metrics,
        "steps_tbler_running": steps_tbler_running,
        "tracks": {
            "drone": np.vstack(drone_traj),
            "ues":   np.stack(ue_traj, axis=0),
            "steps": np.array(steps, dtype=int),
        },
    }



def plot_trajectories_xy_xz(tracks: dict, out_path: Path, title_prefix="Trayectorias",
                            step_stride=5, show_step_labels=False):
    """
    XY + XZ, con títulos separados y leyenda global fuera (lado derecho).
    """
    drone = tracks["drone"]
    ues   = tracks["ues"]
    steps = tracks["steps"]
    T = drone.shape[0]; N = ues.shape[1]

    def _auto_limits_xy(pad=5.0):
        d = drone[:, :2]
        u = ues[:, :, :2].reshape(-1, 2)
        allxy = np.vstack([d, u])
        xmin, ymin = np.nanmin(allxy, axis=0)
        xmax, ymax = np.nanmax(allxy, axis=0)
        return (xmin - pad, xmax + pad, ymin - pad, ymax + pad)

    xmin, xmax, ymin, ymax = _auto_limits_xy(pad=5.0)

    # Usamos subplots_adjust para reservar espacio a la derecha para la leyenda
    fig = plt.figure(figsize=(13.8, 7.6))
    gs  = GridSpec(2, 1, height_ratios=[2, 1], figure=fig)
    # deja margen derecho para la leyenda (≈20%)
    fig.subplots_adjust(left=0.08, right=0.80, top=0.90, bottom=0.08, hspace=0.32)

    ax_xy = fig.add_subplot(gs[0, 0])
    ax_xz = fig.add_subplot(gs[1, 0])

    # ===== XY =====
    ax_xy.set_title(f"{title_prefix} — Planta (XY)", pad=12)
    ax_xy.set_xlabel("X [m]"); ax_xy.set_ylabel("Y [m]")
    ax_xy.set_aspect("equal", adjustable="box")
    ax_xy.set_xlim([xmin, xmax]); ax_xy.set_ylim([ymin, ymax])
    ax_xy.grid(True, ls="--", alpha=0.35)
    ax_xy.set_axisbelow(True)  # grilla debajo

    # Dron
    ax_xy.plot(drone[:,0], drone[:,1], lw=2.0, label="Drone", zorder=2.5)
    ax_xy.scatter(drone[0,0], drone[0,1], s=110, marker="^", label="Drone start", zorder=3)
    ax_xy.scatter(drone[-1,0], drone[-1,1], s=110, marker=">", label="Drone end", zorder=3)
    for k in range(0, T-1, step_stride):
        dx = drone[k+1,0]-drone[k,0]; dy = drone[k+1,1]-drone[k,1]
        ax_xy.arrow(drone[k,0], drone[k,1], dx, dy, length_includes_head=True,
                    head_width=0.8, alpha=0.8, zorder=2.6)

    # UEs
    for i in range(N):
        traj = ues[:, i, :]
        ax_xy.plot(traj[:,0], traj[:,1], lw=1.8, label=f"UE{i}", zorder=2.2)
        ax_xy.scatter(traj[0,0], traj[0,1], s=70, marker="o", zorder=2.7)
        ax_xy.scatter(traj[-1,0], traj[-1,1], s=70, marker="s", zorder=2.7)
        for k in range(0, T-1, step_stride):
            dx = traj[k+1,0]-traj[k,0]; dy = traj[k+1,1]-traj[k,1]
            ax_xy.arrow(traj[k,0], traj[k,1], dx, dy, length_includes_head=True,
                        head_width=0.6, alpha=0.6, zorder=2.4)
        if show_step_labels:
            for k in range(0, T, step_stride):
                ax_xy.text(traj[k,0], traj[k,1], str(steps[k]), fontsize=7, alpha=0.85)

    # No pongas leyenda dentro del eje
    # ax_xy.legend(...)

    # ===== XZ =====
    ax_xz.set_title("Perfil XZ", pad=10)
    ax_xz.set_xlabel("X [m]"); ax_xz.set_ylabel("Z [m]")
    ax_xz.grid(True, ls="--", alpha=0.35)
    ax_xz.set_axisbelow(True)

    ax_xz.plot(drone[:,0], drone[:,2], lw=2.0, label="Drone")
    ax_xz.scatter(drone[0,0], drone[0,2], s=90, marker="^")
    ax_xz.scatter(drone[-1,0], drone[-1,2], s=90, marker=">")
    for i in range(N):
        traj = ues[:, i, :]
        ax_xz.plot(traj[:,0], traj[:,2], lw=1.4, label=f"UE{i}")

    # ===== Leyenda global afuera (derecha) =====
    handles, labels = [], []
    for ax in (ax_xy, ax_xz):
        h, l = ax.get_legend_handles_labels()
        handles += h; labels += l
    # quita duplicados manteniendo orden
    seen=set(); H=[]; L=[]
    for h,l in zip(handles, labels):
        if l not in seen:
            H.append(h); L.append(l); seen.add(l)

    # Colócala fuera, centrada a la derecha
    fig.legend(H, L, loc="center left", bbox_to_anchor=(0.82, 0.5),
               frameon=False, title="Leyenda", borderaxespad=0.0)

    fig.suptitle("Trayectorias — Drone y UEs", y=0.97, fontsize=14)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


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
                se_la = m.get("se_la", np.nan)
                se_sh = m.get("se_shannon", np.nan)
                se_gap = m.get("se_gap_pct", np.nan)
                tbler_step = m.get("tbler", np.nan)
            else:
                ue_id = i
                sinr = prx = se_la = se_sh = se_gap = tbler_step = np.nan

            # TBLER running
            tbler_run = tbler_run_vec[i] if (tbler_run_vec is not None and i < len(tbler_run_vec)) else np.nan

            rows.append({
                "freq_mhz": freq,
                "step": t + 1,  # 1-based para lectura
                "ue_id": ue_id,
                "sinr_eff_db": float(sinr) if sinr is not None else np.nan,
                "prx_dbm": float(prx) if prx is not None else np.nan,
                "se_la": float(se_la) if se_la is not None else np.nan,
                "se_shannon": float(se_sh) if se_sh is not None else np.nan,
                "se_gap_pct": float(se_gap) if se_gap is not None else np.nan,
                "tbler_step": float(tbler_step) if tbler_step is not None else np.nan,
                "tbler_running": float(tbler_run) if tbler_run is not None else np.nan,
            })

    df = pd.DataFrame(rows)
    return df


def plot_metric_per_ue(df_all: pd.DataFrame, metric: str, ylabel: str, out_dir: Path):
    """
    (Sigue existiendo por si quieres PNGs individuales.)
    Con corrección de formato de ejes y anti-solape (offset en X por frecuencia).
    """
    ue_ids = sorted(df_all["ue_id"].dropna().astype(int).unique().tolist())
    freqs = sorted(df_all["freq_mhz"].dropna().unique().tolist())
    label_for = {f: f"{f:.0f} MHz" for f in freqs}

    for ue in ue_ids:
        fig = plt.figure(figsize=(9.5, 5.5))
        ax = plt.gca()

        yvals_all = []
        for j, f in enumerate(freqs):
            dfx = df_all[(df_all["ue_id"] == ue) & (df_all["freq_mhz"] == f)].sort_values("step")
            x = dfx["step"].to_numpy(dtype=float)
            x = x + _X_OFFSETS[j % len(_X_OFFSETS)]  # << anti-solape
            y = dfx[metric].to_numpy(dtype=float)
            yvals_all.append(y[~np.isnan(y)])

            ax.plot(
                x, y,
                marker=_MARKERS[j % len(_MARKERS)],
                linewidth=1.6,
                linestyle=_LINESTY[j % len(_LINESTY)],
                label=label_for[f],
            )

        ax.set_title(f"{metric} por step (UE {ue})")
        ax.set_xlabel("Step")
        ax.set_ylabel(ylabel)
        _axis_format_db(ax)

        # Rango Y razonable (si hay datos)
        flat = np.concatenate([v for v in yvals_all if v.size > 0]) if any(len(v) > 0 for v in yvals_all) else np.array([])
        if flat.size > 0:
            y_min, y_max = np.nanmin(flat), np.nanmax(flat)
            pad = max(1.0, 0.06 * (y_max - y_min if y_max > y_min else 10))
            ax.set_ylim(y_min - pad, y_max + pad)

        ax.legend()
        fig.tight_layout()

        out_file = out_dir / f"{metric}_UE{ue}.png"
        fig.savefig(out_file, dpi=160)
        plt.close(fig)


def _axis_format_db(ax):
    sf = ScalarFormatter(useMathText=False, useOffset=False)
    sf.set_scientific(False)
    ax.yaxis.set_major_formatter(sf)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.grid(True, linestyle="--", alpha=0.35)


def plot_all_metrics_combined(df_all: pd.DataFrame, out_dir: Path):
    """
    Una imagen por UE con 5 subplots (PRx, SINR, SE combinado, TBLER step, TBLER running).
    - Título incluye UE y 'freqs: ...'
    - SE (Shannon vs SE-LA) ocupa el doble de altura (GridSpec con ratios)
    - 'Step' en TODOS los subplots
    - Anti-solape entre frecuencias (offset en X + estilos)
    """
    ue_ids = sorted(df_all["ue_id"].dropna().astype(int).unique().tolist())
    freqs = sorted(df_all["freq_mhz"].dropna().unique().tolist())
    label_for = {f: f"{f:.0f} MHz" for f in freqs}
    freqs_str = ", ".join([label_for[f] for f in freqs])

    for ue in ue_ids:
        # GridSpec: 4 filas, 2 columnas; fila 2 (índice 1) ocupa doble altura
        fig = plt.figure(figsize=(13, 10))
        gs = GridSpec(nrows=4, ncols=2, height_ratios=[1.1, 2.4, 1.3, 0.3], hspace=0.35, wspace=0.28)

        ax_prx   = fig.add_subplot(gs[0, 0])      # PRx
        ax_sinr  = fig.add_subplot(gs[0, 1])      # SINR
        ax_se    = fig.add_subplot(gs[1, :])      # SE combinado (doble altura)
        ax_tbl_s = fig.add_subplot(gs[2, 0])      # TBLER step
        ax_tbl_r = fig.add_subplot(gs[2, 1])      # TBLER running
        ax_dummy = fig.add_subplot(gs[3, :])      # (opcional) libre si quieres algo más
        ax_dummy.axis("off")

        # Para límites con padding
        y_collect = {k: [] for k in ["prx", "sinr", "se", "tbl_s", "tbl_r"]}

        for j, f in enumerate(freqs):
            df_f = df_all[(df_all["ue_id"] == ue) & (df_all["freq_mhz"] == f)].sort_values("step")
            x = df_f["step"].to_numpy(dtype=float)
            x_off = x + _X_OFFSETS[j % len(_X_OFFSETS)]

            # PRx
            y = df_f["prx_dbm"].to_numpy(dtype=float)
            ax_prx.plot(x_off, y, marker=_MARKERS[j % len(_MARKERS)],
                        linestyle=_LINESTY[j % len(_LINESTY)], linewidth=1.6,
                        label=label_for[f])
            y_collect["prx"].append(y[~np.isnan(y)])

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

        # Etiquetas y formato
        ax_prx.set_title("PRx (dBm)")
        ax_sinr.set_title("SINR (dB)")
        ax_se.set_title("SE — Shannon (cota) vs SE-LA (real)")
        ax_tbl_s.set_title("TBLER (por step)")
        ax_tbl_r.set_title("TBLER running")

        ax_prx.set_ylabel("PRx (dBm)")
        ax_sinr.set_ylabel("SINR (dB)")
        ax_se.set_ylabel("SE (b/s/Hz)")
        ax_tbl_s.set_ylabel("TBLER")
        ax_tbl_r.set_ylabel("TBLER")

        # 'Step' en TODOS los subplots solicitados
        for ax in [ax_prx, ax_sinr, ax_se, ax_tbl_s, ax_tbl_r]:
            ax.set_xlabel("Step")

        # Formato y límites
        _axis_format_db(ax_prx)
        _axis_format_db(ax_sinr)
        for key, ax in [("prx", ax_prx), ("sinr", ax_sinr), ("se", ax_se), ("tbl_s", ax_tbl_s), ("tbl_r", ax_tbl_r)]:
            flat = np.concatenate([v for v in y_collect[key] if v.size > 0]) if any(len(v) > 0 for v in y_collect[key]) else np.array([])
            if flat.size > 0:
                y_min, y_max = np.nanmin(flat), np.nanmax(flat)
                pad = max(0.05, 0.08 * (y_max - y_min if y_max > y_min else 1.0))
                ax.set_ylim(y_min - pad, y_max + pad)
            ax.grid(True, linestyle="--", alpha=0.35)

        # Leyenda global (sin duplicados)
        handles, labels = [], []
        for ax in (ax_prx, ax_sinr, ax_se, ax_tbl_s, ax_tbl_r):
            h, l = ax.get_legend_handles_labels()
            handles += h; labels += l
        seen = set(); H, L = [], []
        for h, l in zip(handles, labels):
            if l not in seen:
                H.append(h); L.append(l); seen.add(l)
        fig.legend(H, L, loc="lower center", ncol=min(6, max(2, len(L)//2)),
                   frameon=False, bbox_to_anchor=(0.5, 0.02))

        fig.suptitle(f"UE {ue} — freqs: {freqs_str}", fontsize=14, y=0.995)
        fig.tight_layout(rect=[0.04, 0.08, 0.98, 0.97])

        out_file = out_dir / f"UE{ue}_all_metrics.png"
        fig.savefig(out_file, dpi=180, bbox_inches="tight")
        plt.close(fig)


def plot_all_metrics_single_freq(df_all: pd.DataFrame, freq_mhz: float, out_dir: Path):
    """
    Una imagen por UE pero mostrando SOLO la frecuencia 'freq_mhz'.
    Conserva el layout (SE con doble altura) y 'Step' en todos los subplots.
    Sin offsets (no hacen falta con una sola frecuencia).
    """
    df_f_all = df_all[np.isclose(df_all["freq_mhz"], freq_mhz)].copy()
    if df_f_all.empty:
        print(f"[WARN] No hay datos para {freq_mhz} MHz")
        return

    ue_ids = sorted(df_f_all["ue_id"].dropna().astype(int).unique().tolist())
    label = f"{freq_mhz:.0f} MHz"

    for ue in ue_ids:
        fig = plt.figure(figsize=(13, 10))
        gs = GridSpec(nrows=4, ncols=2, height_ratios=[1.1, 2.4, 1.3, 0.3], hspace=0.35, wspace=0.28)
        ax_prx   = fig.add_subplot(gs[0, 0])
        ax_sinr  = fig.add_subplot(gs[0, 1])
        ax_se    = fig.add_subplot(gs[1, :])
        ax_tbl_s = fig.add_subplot(gs[2, 0])
        ax_tbl_r = fig.add_subplot(gs[2, 1])
        ax_dummy = fig.add_subplot(gs[3, :]); ax_dummy.axis("off")

        df_f = df_f_all[(df_f_all["ue_id"] == ue)].sort_values("step")
        x = df_f["step"].to_numpy(dtype=float)

        # PRx
        y = df_f["prx_dbm"].to_numpy(dtype=float)
        ax_prx.plot(x, y, marker="o", linestyle="-", linewidth=1.8, label=label)

        # SINR
        y = df_f["sinr_eff_db"].to_numpy(dtype=float)
        ax_sinr.plot(x, y, marker="o", linestyle="-", linewidth=1.8, label=label)

        # SE combinado
        y_sh = df_f["se_shannon"].to_numpy(dtype=float)
        y_la = df_f["se_la"].to_numpy(dtype=float)
        ax_se.plot(x, y_sh, marker="o", linestyle="-",  linewidth=1.9, label=f"{label} · Shannon")
        ax_se.plot(x, y_la, marker="s", linestyle="--", linewidth=1.7, label=f"{label} · SE-LA")

        # TBLER step
        y = df_f["tbler_step"].to_numpy(dtype=float)
        ax_tbl_s.plot(x, y, marker="o", linestyle="-", linewidth=1.8, label=label)

        # TBLER running
        y = df_f["tbler_running"].to_numpy(dtype=float)
        ax_tbl_r.plot(x, y, marker="o", linestyle="-", linewidth=1.8, label=label)

        # Títulos / labels / formato
        ax_prx.set_title("PRx (dBm)");       ax_prx.set_ylabel("PRx (dBm)")
        ax_sinr.set_title("SINR (dB)");      ax_sinr.set_ylabel("SINR (dB)")
        ax_se.set_title("SE — Shannon (cota) vs SE-LA (real)"); ax_se.set_ylabel("SE (b/s/Hz)")
        ax_tbl_s.set_title("TBLER (por step)"); ax_tbl_s.set_ylabel("TBLER")
        ax_tbl_r.set_title("TBLER running");   ax_tbl_r.set_ylabel("TBLER")

        for ax in [ax_prx, ax_sinr, ax_se, ax_tbl_s, ax_tbl_r]:
            ax.set_xlabel("Step")
            ax.grid(True, linestyle="--", alpha=0.35)

        _axis_format_db(ax_prx); _axis_format_db(ax_sinr)

        # Límites con padding
        def _pad(ax, yvals):
            yv = yvals[~np.isnan(yvals)]
            if yv.size:
                y_min, y_max = np.min(yv), np.max(yv)
                pad = max(0.05, 0.08 * (y_max - y_min if y_max > y_min else 1.0))
                ax.set_ylim(y_min - pad, y_max + pad)

        _pad(ax_prx,  df_f["prx_dbm"].to_numpy(dtype=float))
        _pad(ax_sinr, df_f["sinr_eff_db"].to_numpy(dtype=float))
        _pad(ax_se,   np.concatenate([y_sh[~np.isnan(y_sh)], y_la[~np.isnan(y_la)]]))
        _pad(ax_tbl_s, df_f["tbler_step"].to_numpy(dtype=float))
        _pad(ax_tbl_r, df_f["tbler_running"].to_numpy(dtype=float))

        # Leyenda global
        handles, labels = [], []
        for ax in (ax_prx, ax_sinr, ax_se, ax_tbl_s, ax_tbl_r):
            h, l = ax.get_legend_handles_labels()
            handles += h; labels += l
        seen = set(); H, L = [], []
        for h, l in zip(handles, labels):
            if l not in seen:
                H.append(h); L.append(l); seen.add(l)
        fig.legend(H, L, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.02))

        fig.suptitle(f"UE {ue} — freq: {label}", fontsize=14, y=0.995)
        fig.tight_layout(rect=[0.04, 0.08, 0.98, 0.97])

        out_file = out_dir / f"UE{ue}_all_metrics_{int(freq_mhz)}MHz.png"
        fig.savefig(out_file, dpi=180, bbox_inches="tight")
        plt.close(fig)


def main():
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


    # plots métricas
    plot_all_metrics_combined(df_all, OUT_DIR)


    #plot_metric_per_ue(df_all, metric="prx_dbm",       ylabel="PRx (dBm)",            out_dir=OUT_DIR)
    #plot_metric_per_ue(df_all, metric="sinr_eff_db",   ylabel="SINR efectivo (dB)",   out_dir=OUT_DIR)
    #plot_metric_per_ue(df_all, metric="se_la",         ylabel="SE (LA) [bits/s/Hz]",  out_dir=OUT_DIR)
    #plot_metric_per_ue(df_all, metric="se_shannon",    ylabel="SE (Shannon) [b/s/Hz]",out_dir=OUT_DIR)
    #plot_metric_per_ue(df_all, metric="tbler_step",    ylabel="TBLER (por step)",     out_dir=OUT_DIR)
    #plot_metric_per_ue(df_all, metric="tbler_running", ylabel="TBLER running",        out_dir=OUT_DIR)


 
    # plots por frecuencia
    for f in FREQS_MHZ:
        plot_all_metrics_single_freq(df_all, f, OUT_DIR)

    # mapas XY+XZ
    for r in runs:
        fmhz = r["freq_mhz"]
        out_traj = OUT_DIR / f"traj_{int(fmhz)}MHz.png"
        title = f"SCENE={SCENE} — freq={fmhz:.0f} MHz — DRONE_START={DRONE_START}"
        plot_trajectories_xy_xz(r["tracks"], out_path=out_traj, title_prefix=title,
                                step_stride=5, show_step_labels=False)

    print(f"[DONE] Imágenes en: {OUT_DIR}")


if __name__ == "__main__":
    main()
