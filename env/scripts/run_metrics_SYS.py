# === Bootstrap sys.path a la raíz del proyecto (dos niveles arriba) ===
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Backend interactivo para ver gráficos (usa TkAgg o Qt5Agg)
import os
os.environ.pop("MPLBACKEND", None)
import matplotlib
matplotlib.use("TkAgg", force=True)

import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

from env.environment.gymnasium_env import DroneEnv  # tu env

# =============== Configuración del escenario ===============
SCENE = "simple_street_canyon_with_cars"
DRONE_START = (0.0, 0.0, 10.0)
RX_POSITIONS = [
    (-50.0, 0.0, 1.5),
    (0.0,   30.0, 1.5),
    (20.0, -30.0, 1.5),
    (80.0,  40.0, 1.5),
    (50.0,   0.0, 1.5),
    (90.0, -55.0, 1.5),
]

MAX_STEPS   = 30     # <<==== 100 steps
PRINT_EVERY = 10      # imprime cada 10 steps (pon 1 si quieres cada step)

OUTDIR = Path(project_root) / "outputs" / "runs" / "2"
OUTDIR.mkdir(parents=True, exist_ok=True)
ts = datetime.now().strftime("%Y%m%d-%H%M%S")
xlsx_path = OUTDIR / f"simulacion_metricas.xlsx"

def maybe_print(step, total, when=10):
    if when <= 1:
        print(f"[{step:3d}/{total}] ejecutado")
    else:
        if step == 1 or step % when == 0 or step == total:
            print(f"[{step:3d}/{total}] ejecutado")

def plot_metrics(df: pd.DataFrame, prefix: str):
    """ Genera y guarda 4 figuras: SINR, SE, BLER y Goodput por usuario """
    ue_ids = sorted(df["ue_id"].unique())

    # --- 1) SINR efectivo ---
    plt.figure(figsize=(10, 5))
    for u in ue_ids:
        d = df[df.ue_id == u]
        plt.plot(d["step"], d["sinr_eff_db"], marker="o", linestyle="-", label=f"UE {u}")
    plt.title("SINR efectivo por step")
    plt.xlabel("Step")
    plt.ylabel("SINR efectivo [dB]")
    plt.grid(True, alpha=0.3)
    plt.legend()
    sinr_png = OUTDIR / f"{prefix}_sinr.png"
    plt.savefig(sinr_png, dpi=140, bbox_inches="tight")

    # --- 2) SE: OLLA vs Shannon ---
    plt.figure(figsize=(10, 5))
    for u in ue_ids:
        d = df[df.ue_id == u]
        plt.plot(d["step"], d["se_la"], marker="o", linestyle="-", label=f"SE OLLA - UE {u}")
    for u in ue_ids:
        d = df[df.ue_id == u]
        plt.plot(d["step"], d["se_shannon"], marker="x", linestyle="--", label=f"SE Shannon - UE {u}")
    plt.title("Eficiencia espectral: OLLA vs Shannon")
    plt.xlabel("Step")
    plt.ylabel("bps/Hz")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2)
    se_png = OUTDIR / f"{prefix}_se.png"
    plt.savefig(se_png, dpi=140, bbox_inches="tight")

    # --- 3) BLER ---
    plt.figure(figsize=(10, 5))
    for u in ue_ids:
        d = df[df.ue_id == u].copy()
        # Si en tu pipeline BLER=2 significa "no agendado", lo enmascaramos:
        d.loc[d["bler"] >= 1.5, "bler"] = np.nan
        plt.plot(d["step"], d["bler"], marker="o", linestyle="-", label=f"UE {u}")
    plt.axhline(0.1, linestyle="--", linewidth=1.0)  # objetivo BLER
    plt.title("BLER por step (línea punteada = objetivo 0.1)")
    plt.xlabel("Step")
    plt.ylabel("BLER")
    plt.ylim(-0.02, 0.3)
    plt.grid(True, alpha=0.3)
    plt.legend()
    bler_png = OUTDIR / f"{prefix}_bler.png"
    plt.savefig(bler_png, dpi=140, bbox_inches="tight")

    # --- 4) Goodput (bits decodificados por step) ---
    plt.figure(figsize=(10, 5))
    for u in ue_ids:
        d = df[df.ue_id == u]
        plt.plot(d["step"], d["num_decoded_bits"], marker="o", linestyle="-", label=f"UE {u}")
    plt.title("Bits decodificados (goodput) por step")
    plt.xlabel("Step")
    plt.ylabel("Bits")
    plt.grid(True, alpha=0.3)
    plt.legend()
    goodput_png = OUTDIR / f"{prefix}_goodput.png"
    plt.savefig(goodput_png, dpi=140, bbox_inches="tight")

    plt.show()
    print(f"Figuras guardadas en:\n- {sinr_png}\n- {se_png}\n- {bler_png}\n- {goodput_png}")

if __name__ == "__main__":
    env = DroneEnv(
        scene_name=SCENE,
        max_steps=MAX_STEPS,
        drone_start=DRONE_START,
        rx_positions=RX_POSITIONS if RX_POSITIONS else None,
        antenna_mode="ISO",  # "ISO" o "SECTOR3_3GPP"
        # render_mode="human",  # si quieres ventana RT
    )

    obs, info = env.reset(seed=0)
    done, trunc = False, False

    metrics_history = []
    while not (done or trunc):
        # Política “quieta”: sin mover el dron (ajusta si quieres)
        a = [0.0, 0.0, 0.0]
        obs, rew, done, trunc, info = env.step(a)

        # Guardar métricas por-UE para el step actual
        for ue in info["ue_metrics"]:
            row = {"step": env.step_count}
            row.update(ue)
            metrics_history.append(row)

        maybe_print(env.step_count, MAX_STEPS, PRINT_EVERY)

    env.close()

    # ---- DataFrame + Excel ----
    df = pd.DataFrame(metrics_history)
    # opcional: ordenar columnas
    cols = ["step","ue_id","sinr_eff_db","prx_dbm","se_la","se_shannon",
            "num_decoded_bits","goodput","bler","mcs_index"]
    df = df[[c for c in cols if c in df.columns]]

    df.to_excel(xlsx_path, index=False)
    print(f"Excel generado: {xlsx_path}")

    # ---- Gráficos ----
    plot_metrics(df, prefix=f"run_")
