# run_sfm_test_v2.py
# -*- coding: utf-8 -*-
import os
import sys
from pathlib import Path
import numpy as np
import matplotlib
from datetime import datetime
import matplotlib.animation as animation
import torch  # Necesario para manejar tensores si aparecen

# Backend no interactivo
os.environ["MPLBACKEND"] = "Agg"
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# === Bootstrap sys.path ===
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from env.environment.gymnasium_env import DroneEnv

# ========= Configuración =========
SCENE = "simple_street_canyon_with_cars"
DRONE_START = (0.0, 0.0, 10.0)
MAX_STEPS = 300
FREQS_MHZ = [3500.0]

# Posiciones desafiantes (Tienen que esquivar el obstáculo central)
RX_POSITIONS = [
    #Prueba original con 3 receptores
    #(20.0, -30.0, 1.5),
    #(50.0,    0.0, 1.5),
    #(-90, -55, 1.5),

    #Prueba de Colisión entre 2 receptores (sin colisión)
    #(-40.0, 0.0, 1.5),  #UE0
    #(-20.0, 0.0, 1.5),  #UE1

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

    #Prueba de Colisión entre 7 receptores (sin colisión)
    #Metas Grupo 1
    (-10.0, 1.0, 1.5),   # UE0
    (-10.0, -1.0, 1.5),  # UE1
    (-12.0, 0.5, 1.5),   # UE2
    (-12.0, -0.5, 1.5),  # UE3

    #Metas Grupo 2
    (-40.0, 0.0, 1.5),   # UE4
    (-40.0, 1.0, 1.5),   # UE5
    (-40.0, -1.0, 1.5),  # UE6
]

# Carpeta de salida
RUN_TAG = datetime.now().strftime("%Y%m%d-%H%M%S")
OUT_DIR = Path(project_root) / "Pruebas SFM Slicer" / f"SFM_{RUN_TAG}"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# === Funciones Auxiliares (CORREGIDAS) ===
def _vec3_to_np(p):
    """
    Convierte ROBUSTAMENTE cualquier tipo de vector (Tensor, List, DrJit) a Numpy.
    """
    try:
        # Intento 1: Conversión directa a Numpy (Funciona para Listas, Tuplas, Tensores TF y Torch CPU)
        val = np.array(p, dtype=float)

        # Si quedó con shape (1, 3) o similar, lo aplanamos
        if val.size == 3:
            return val.reshape(3)
    except Exception:
        pass

    # Intento 2: Si es un objeto con x,y,z (Mitsuba/DrJit antiguo)
    try:
        return np.array([float(p.x), float(p.y), float(p.z)], dtype=float)
    except Exception:
        pass

    # Fallback si todo falla (pero avisa)
    # print(f"[WARNING] No se pudo convertir vector: {type(p)}")
    return np.array([0., 0., 0.])


def _get_drone_xyz(rt):
    if hasattr(rt, "tx") and rt.tx: return _vec3_to_np(rt.tx.position)
    return np.array([0., 0., 10.])


def _get_rx_positions_xyz(rt):
    pos = []
    for rx in rt.rx_list:
        pos.append(_vec3_to_np(rx.position))
    return np.vstack(pos).astype(float)


# === 1. PLOT ESTÁTICO (FINAL: Con etiquetas de texto) ===
def plot_trajectories_xy_xz(tracks, obstacles, out_path, title_prefix="Trayectorias"):
    drone = tracks["drone"]
    ues = tracks["ues"]
    N = ues.shape[1]
    rx_labels = [f"UE{i}" for i in range(N)]

    # 1. Calcular límites globales para XY
    all_xy = np.vstack([drone[:, :2], ues[:, :, :2].reshape(-1, 2)])
    if obstacles:
        try:
            obs_stack = np.vstack(obstacles)
            all_xy = np.vstack([all_xy, obs_stack])
        except:
            pass

    pad = 5.0
    xmin, xmax = np.nanmin(all_xy[:, 0]) - pad, np.nanmax(all_xy[:, 0]) + pad
    ymin, ymax = np.nanmin(all_xy[:, 1]) - pad, np.nanmax(all_xy[:, 1]) + pad

    # 2. Configuración de Figura
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(2, 1, height_ratios=[3, 1.2], figure=fig)
    fig.subplots_adjust(left=0.08, right=0.82, top=0.92, bottom=0.08, hspace=0.35)

    ax_xy = fig.add_subplot(gs[0, 0])
    ax_xz = fig.add_subplot(gs[1, 0])

    # --- PLANTA (XY) ---
    ax_xy.set_title(f"{title_prefix} — Planta (XY)", pad=12, fontsize=14, weight='bold')
    ax_xy.set_xlabel("X [m]", fontsize=12)
    ax_xy.set_ylabel("Y [m]", fontsize=12)
    ax_xy.set_aspect("equal", adjustable='datalim')
    ax_xy.set_xlim([xmin, xmax])
    ax_xy.set_ylim([ymin, ymax])
    ax_xy.grid(True, ls="--", alpha=0.35)

    # Obstáculos XY
    if obstacles:
        for obs in obstacles:
            ax_xy.scatter(obs[:, 0], obs[:, 1], s=4, c='#404040', marker='s', alpha=0.3, linewidths=0)

    # Dron XY
    ax_xy.plot(drone[:, 0], drone[:, 1], lw=3.0, label="Drone", zorder=2.5, color='tab:blue')
    ax_xy.scatter(drone[0, 0], drone[0, 1], s=150, marker="^", label="Drone start", zorder=3, edgecolors='k',
                  color='tab:blue')

    # UEs XY
    colors = matplotlib.colormaps['tab10']
    for i in range(N):
        c = colors(i)
        traj = ues[:, i, :]

        # Trayectoria
        ax_xy.plot(traj[:, 0], traj[:, 1], lw=3, color=c, label=rx_labels[i], zorder=2.2, alpha=0.9)

        # Inicio (Círculo)
        ax_xy.scatter(traj[0, 0], traj[0, 1], s=120, marker="o", color=c, zorder=2.8, edgecolors='white')

        # Fin (Cuadrado)
        ax_xy.scatter(traj[-1, 0], traj[-1, 1], s=120, marker="s", color=c, zorder=2.8, edgecolors='white')

        # --- AQUÍ ESTÁ EL CAMBIO: ETIQUETAS DE TEXTO ---
        # Ponemos el nombre del UE un poco arriba y a la derecha de su punto de inicio
        ax_xy.text(traj[0, 0] + 1.5, traj[0, 1] + 1.5, rx_labels[i],
                   fontsize=11, weight="bold", color=c, zorder=5)

    # --- PERFIL (XZ) ---
    ax_xz.set_title("Perfil XZ (Elevación)", pad=10, fontsize=14, weight='bold')
    ax_xz.set_xlabel("X [m]", fontsize=12)
    ax_xz.set_ylabel("Z [m]", fontsize=12)
    ax_xz.grid(True, ls="--", alpha=0.35)

    ax_xz.set_xlim([xmin, xmax])
    ax_xz.set_ylim(0, 18)
    ax_xz.set_aspect('auto')

    # Dron XZ
    ax_xz.plot(drone[:, 0], drone[:, 2], lw=3.0, label="Drone", color='tab:blue')
    ax_xz.scatter(drone[0, 0], drone[0, 2], s=100, marker="^", color='tab:blue', edgecolors='k')

    # UEs XZ
    for i in range(N):
        ax_xz.plot(ues[:, i, 0], ues[:, i, 2], lw=2.5, color=colors(i), label=rx_labels[i], alpha=0.8)
        ax_xz.scatter(ues[0, i, 0], ues[0, i, 2], s=80, marker="o", color=colors(i), edgecolors='white')

    # --- LEYENDA UNIFICADA A LA DERECHA ---
    handles_xy, labels_xy = ax_xy.get_legend_handles_labels()
    unique_labels = dict(zip(labels_xy, handles_xy))

    from matplotlib.lines import Line2D
    obst_proxy = Line2D([0], [0], marker='s', color='w', label='Obstáculos',
                        markerfacecolor='#404040', markersize=10, alpha=0.5)

    final_handles = [unique_labels['Drone']] + [unique_labels[l] for l in rx_labels if l in unique_labels] + [
        obst_proxy]
    final_labels = ['Drone'] + [l for l in rx_labels if l in unique_labels] + ['Obstáculos']

    fig.legend(final_handles, final_labels,
               loc="center left", bbox_to_anchor=(0.83, 0.5),
               title="Leyenda", title_fontsize=12, fontsize=11,
               frameon=True, fancybox=True, shadow=True, borderpad=1)

    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# === 2. GENERADOR DE GIF (CON LEYENDA EXTERNA) ===
def make_gif(tracks, obstacles, out_path, fps=20):
    ues = tracks["ues"]
    T, N, _ = ues.shape

    # Figura más ancha para la leyenda
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.subplots_adjust(right=0.75) # Dejar espacio a la derecha

    ax.set_aspect('equal')
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_title("Simulación Dinámica (Sionna + SocialForce)")
    ax.grid(True, alpha=0.3)

    # Obstáculos fijos
    if obstacles:
        for obs in obstacles:
            ax.scatter(obs[:, 0], obs[:, 1], s=1, c='black', marker='.', alpha=0.2)

    # Límites dinámicos
    all_xy = ues[:, :, :2].reshape(-1, 2)
    if obstacles:
        try:
            obs_stack = np.vstack(obstacles)
            all_xy = np.vstack([all_xy, obs_stack])
        except: pass
    pad = 5
    ax.set_xlim(np.min(all_xy[:, 0]) - pad, np.max(all_xy[:, 0]) + pad)
    ax.set_ylim(np.min(all_xy[:, 1]) - pad, np.max(all_xy[:, 1]) + pad)

    # Colores y elementos dinámicos
    cmap = matplotlib.colormaps['tab10']
    colors = [cmap(i) for i in range(N)]
    rx_labels = [f"UE{i}" for i in range(N)]

    start_pos = ues[0, :, :2]
    scats = ax.scatter(start_pos[:, 0], start_pos[:, 1], s=120, c=colors, zorder=5, edgecolors='white')
    trails = [ax.plot([], [], '-', lw=2, color=colors[i], alpha=0.6, label=rx_labels[i])[0] for i in range(N)]

    # --- LEYENDA EXTERNA ---
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), title="Receptores")

    def update(frame):
        current_pos = ues[frame, :, :2]
        scats.set_offsets(current_pos)
        start = max(0, frame - 30) # Trail más largo
        for i, trail in enumerate(trails):
            trail.set_data(ues[start:frame + 1, i, 0], ues[start:frame + 1, i, 1])
        return scats, *trails

    frames_idx = range(0, T, 2)
    ani = animation.FuncAnimation(fig, update, frames=frames_idx, interval=50, blit=True)
    ani.save(out_path, writer='pillow', fps=fps)
    plt.close(fig)


# === LOOP DE SIMULACIÓN ===
def run_episode(freq_mhz: float) -> dict:
    print(f"--- Iniciando Episodio {freq_mhz} MHz ---")

    env = DroneEnv(
        render_mode=None,
        scene_name=SCENE,
        max_steps=MAX_STEPS,
        drone_start=DRONE_START,
        rx_positions=RX_POSITIONS,
        rx_goals=RX_GOALS,
        frequency_mhz=freq_mhz,
        run_metrics=False
    )

    # 1. Extraer los obstáculos REALES (Slicer)
    # Esto confirma que el entorno los tiene cargados
    try:
        sfm_obstacles_torch = env.sfm_sim.ped_space.space
        obstacles_np = [o.numpy() for o in sfm_obstacles_torch]
        print(f"[INFO] Visualizador: {len(obstacles_np)} grupos de obstáculos detectados.")
    except AttributeError:
        print("[WARNING] No se pudieron leer los obstáculos del simulador (¿Falta self.sfm_sim.ped_space?)")
        obstacles_np = []

    obs, info = env.reset(seed=0)
    drone_traj, ue_traj, steps = [], [], []

    # Imagen de referencia (Mapa de calor)
    out_img = OUT_DIR / f"radio_map_ref.png"
    if not out_img.exists():
        env.rt.render_scene_to_file(filename=str(out_img), with_radio_map=True)

    print("Simulando pasos...")
    t = 0

    # DEBUG: Verificamos posición inicial
    first_pos = _get_rx_positions_xyz(env.rt)
    print(f"[DEBUG] Posiciones iniciales leídas por el visualizador:\n{first_pos}")

    while t < MAX_STEPS:
        # Guardar posiciones
        drone_traj.append(_get_drone_xyz(env.rt).copy())
        ue_traj.append(_get_rx_positions_xyz(env.rt).copy())
        steps.append(t)

        # Paso de simulación
        obs, rew, done, trunc, info = env.step(np.array([0, 0, 0], dtype=float))
        t += 1
        if done or trunc: break

    env.close()

    return {
        "freq_mhz": freq_mhz,
        "obstacles": obstacles_np,
        "tracks": {
            "drone": np.vstack(drone_traj),
            "ues": np.stack(ue_traj, axis=0),
            "steps": np.array(steps, dtype=int),
        },
    }


def main():
    print(f"[INFO] Guardando resultados en: {OUT_DIR}")

    for f in FREQS_MHZ:
        r = run_episode(f)
        tracks = r["tracks"]
        obstacles = r["obstacles"]

        # 1. Plot Estático
        out_traj = OUT_DIR / f"traj_static_{int(f)}MHz.png"
        plot_trajectories_xy_xz(tracks, obstacles, out_path=out_traj, title_prefix=f"SFM Slicer {f:.0f}MHz")
        print(f"Imagen guardada: {out_traj}")

        # 2. GIF
        out_gif = OUT_DIR / f"animacion_{int(f)}MHz.gif"
        print("Generando GIF...")
        make_gif(tracks, obstacles, out_path=out_gif)
        print(f"GIF guardado: {out_gif}")

    print(f"[DONE] Pruebas finalizadas.")


if __name__ == "__main__":
    main()