# -*- coding: utf-8 -*-
from __future__ import annotations
import datetime
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas  # para rgb_array
import sys
from pathlib import Path

# (Se mantiene para no romper ejecuciones fuera del paquete)
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Proyecto
from .sionna import SionnaRT
from .dron import Dron
from .receptores import ReceptoresManager, Receptor


class DroneEnv(gym.Env):
    """Entorno Gymnasium con Sionna RT (sin imagen de fondo)."""
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 15}

    def __init__(
        self,
        rx_positions: list[tuple[float, float, float]] | None = None,
        frequency_mhz: float = 3500.0,
        tx_power_dbm: float = 30.0,
        noise_figure_db: float = 7.0,
        bandwidth_hz: float = 20e6,
        scene_name: str = "munich",
        antenna_mode: str = "ISO",
        max_steps: int = 400,
        render_mode: str | None = None,
        drone_start: tuple[float, float, float] = (10.0, 0.0, 20.0),
    ):
        super().__init__()
        assert render_mode in (None, "human", "rgb_array"), \
            "render_mode debe ser None, 'human' o 'rgb_array'"
        self.render_mode = render_mode

        self._start = drone_start
        self.max_steps = int(max_steps)
        self.step_count = 0

        # Receptores por defecto en anillo si no se pasan
        if rx_positions is None:
            r, n = 100.0, 8
            rx_positions = [
                (r*np.cos(2*np.pi*k/n), r*np.sin(2*np.pi*k/n), 1.5) for k in range(n)
            ]
        self.receptores = ReceptoresManager([Receptor(*p) for p in rx_positions])

        
        # Mundo Sionna RT
        self.rt = SionnaRT(
            antenna_mode=antenna_mode,
            frequency_mhz=frequency_mhz,
            tx_power_dbm=tx_power_dbm,
            noise_figure_db=noise_figure_db,
            bandwidth_hz=bandwidth_hz,
            scene_name=scene_name,
        )
        self.rt.build_scene()
        self.rt.attach_receivers(self.receptores.positions_xyz())

        # Dron / spaces
        self.dron = Dron(start_xyz=self._start)
        self.action_space = spaces.Box(low=-5.0, high=5.0, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-1e9, high=1e9, shape=(3 + self.receptores.n,), dtype=np.float32
        )

        # Estado de render
        self._fig = None
        self._ax = None
        self._canvas = None

        self._drone_sc = None
        self._sc = None
        self._texts = []
        self._cbar = None

    # ================= Gym API =================
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.step_count = 0

        self.dron = Dron(start_xyz=self._start)
        self.rt.move_tx(self.dron.pos)

        prx = self.rt.compute_prx_dbm()
        obs = np.concatenate([self.dron.pos, prx]).astype(np.float32)
        info = {}

        # Reset de artistas
        self._drone_sc = None
        self._sc = None
        self._texts = []
        self._cbar = None

        return obs, info

    def step(self, action: np.ndarray):
        self.step_count += 1

        # Movimiento del dron
        self.dron.step_delta(action)
        self.rt.move_tx(self.dron.pos)

        #Movimiento de personas

        # Señal / SNR
        prx = self.rt.compute_prx_dbm()
        snr = self.rt.compute_snr_db(prx)
        reward = float(np.mean(snr))

        terminated = False
        truncated = self.step_count >= self.max_steps

        obs = np.concatenate([self.dron.pos, prx]).astype(np.float32)
        info = {"snr_db": snr, "prx_dbm": prx}

        if self.render_mode == "human":
            self._render_to_figure()
        elif self.render_mode == "rgb_array":
            frame = self._render_to_array()
            info["frame"] = frame

        return obs, reward, terminated, truncated, info

    # ================= Render helpers =================
    def _ensure_figure(self):
        import matplotlib.pyplot as plt
        if self._fig is None or self._ax is None:
            self._fig, self._ax = plt.subplots(figsize=(6, 6), dpi=100)
            self._canvas = FigureCanvas(self._fig)
            self._ax.set_aspect("equal", adjustable="box")
            self._ax.set_title("Cobertura (PRx dBm)")
            self._ax.set_xlabel("x [m]")
            self._ax.set_ylabel("y [m]")
            self._ax.grid(True, alpha=0.3)
            if self.render_mode == "human":
                plt.ion()
                plt.show(block=False)

    def _render_common(self):
        import matplotlib.pyplot as plt  # para plt.pause
        prx = self.rt.compute_prx_dbm()
        prx = np.asarray(prx, dtype=float).reshape(-1) 
        rx = self.receptores.positions_xyz()

        if self._sc is None:
            self._drone_sc = self._ax.scatter(
                [self.dron.pos[0]], [self.dron.pos[1]], s=120, marker="^"
            )
            self._sc = self._ax.scatter(rx[:, 0], rx[:, 1], s=80, c=prx, cmap="viridis")
            self._texts = [
                self._ax.text(x + 1, y + 1, f"{v:.1f} dBm", fontsize=8)
                for (x, y, _), v in zip(rx, prx)
            ]
            if self._cbar is None:
                self._cbar = self._fig.colorbar(self._sc, ax=self._ax, label="PRx [dBm]")
            else:
                self._cbar.update_normal(self._sc)
            return

        self._drone_sc.set_offsets([[self.dron.pos[0], self.dron.pos[1]]])
        self._sc.set_offsets(rx[:, :2])
        self._sc.set_array(prx)
        for t, (x, y, _), v in zip(self._texts, rx, prx):
            t.set_position((x + 1, y + 1))
            t.set_text(f"{float(v):.1f} dBm")
        self._cbar.update_normal(self._sc)

    def _render_to_figure(self):
        import matplotlib.pyplot as plt
        self._ensure_figure()
        self._render_common()
        self._fig.canvas.draw()
        self._fig.canvas.flush_events()
        plt.pause(1.0 / max(1, self.metadata.get("render_fps", 15)))

    def _render_to_array(self) -> np.ndarray:
        self._ensure_figure()
        self._render_common()
        self._canvas.draw()
        w, h = self._fig.get_size_inches() * self._fig.dpi
        w, h = int(w), int(h)
        argb = np.frombuffer(self._canvas.tostring_argb(), dtype=np.uint8).reshape((h, w, 4))
        return argb[:, :, 1:4].copy()  # ARGB -> RGB

    def render(self):
        if self.render_mode == "human":
            self._render_to_figure()
        elif self.render_mode == "rgb_array":
            return self._render_to_array()

    def close(self):
        import matplotlib.pyplot as plt
        if self._fig is not None:
            plt.close(self._fig)
        self._fig = self._ax = self._canvas = self._cbar = None


    def render_dual_snapshot(self,
                            prx_left,
                            prx_right,
                            title_left="PRx teórico (dBm)",
                            title_right="PRx real Sionna (dBm)",
                            show_values_in_labels=True):
        import numpy as np
        import matplotlib.pyplot as plt

        prx_left  = np.asarray(prx_left, dtype=float).reshape(-1)
        prx_right = np.asarray(prx_right, dtype=float).reshape(-1)
        rx = self.receptores.positions_xyz()
        drone_xy = (self.dron.pos[0], self.dron.pos[1])
        d = np.asarray(self.rt.compute_tx_rx_distances(), dtype=float).reshape(-1)

        # === Banner RF (f, Pt, NF, B) ===
        try:
            fc_ghz = float(getattr(self.rt, "freq_hz", np.nan)) / 1e9
        except Exception:
            fc_ghz = np.nan
        try:
            pt_dbm = float(getattr(self, "tx_power_dbm", np.nan))
            if np.isnan(pt_dbm):
                pt_dbm = float(self.rt._total_tx_power_dbm())
        except Exception:
            pt_dbm = float(self.rt._total_tx_power_dbm())
        try:
            nf_db = float(getattr(self, "noise_figure_db", np.nan))
        except Exception:
            nf_db = np.nan
        try:
            bw_mhz = float(getattr(self, "bandwidth_hz", np.nan)) / 1e6
        except Exception:
            bw_mhz = np.nan

        rf_str = "RF: "
        rf_parts = []
        rf_parts.append(f"f={fc_ghz:.3f} GHz" if not np.isnan(fc_ghz) else "f=N/A")
        rf_parts.append(f"Pt={pt_dbm:.1f} dBm" if not np.isnan(pt_dbm) else "Pt=N/A")
        rf_parts.append(f"NF={nf_db:.1f} dB" if not np.isnan(nf_db) else "NF=N/A")
        rf_parts.append(f"B={bw_mhz:.1f} MHz" if not np.isnan(bw_mhz) else "B=N/A")
        rf_str += " | ".join(rf_parts)

        # Escala común para comparación justa
        vmin = float(np.nanmin([prx_left.min(), prx_right.min()]))
        vmax = float(np.nanmax([prx_left.max(), prx_right.max()]))

        # Figura 1x3: izq (teo), centro (tabla abajo), der (real)
        fig = plt.figure(figsize=(15, 6), dpi=110)
        gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 0.8, 1.0], wspace=0.15)
        axL = fig.add_subplot(gs[0, 0])
        axC = fig.add_subplot(gs[0, 1])
        axR = fig.add_subplot(gs[0, 2])

        # --- Panel izquierdo: Teórico ---
        scL = axL.scatter(rx[:,0], rx[:,1], c=prx_left, s=80, cmap="viridis", vmin=vmin, vmax=vmax)
        axL.scatter([drone_xy[0]], [drone_xy[1]], marker="^", s=150, edgecolors="k", facecolors="none", label="Dron")
        for i, (x, y, _) in enumerate(rx):
            label = f"Rx{i}" if not show_values_in_labels else f"Rx{i}\n{prx_left[i]:.1f} dBm"
            axL.text(x + 1, y + 1, label, fontsize=8, weight="bold")
        axL.set_aspect("equal", adjustable="box")
        axL.set_title(title_left)
        # Banner RF debajo del título
        axL.text(0.5, 1.3, rf_str, transform=axL.transAxes, ha="center", va="bottom", fontsize=9)
        axL.set_xlabel("x [m]"); axL.set_ylabel("y [m]")
        axL.grid(True, alpha=0.3)
        fig.colorbar(scL, ax=axL, label="dBm")
        axL.legend(loc="upper right")

        # --- Panel central: “tabla” abajo (distancia + PRx teo/real) ---
        axC.axis("off")
        lines = [
            f"Rx{i:02d}  d={d[i]:6.2f} m   Teo={prx_left[i]:7.2f} dBm   Real={prx_right[i]:7.2f} dBm"
            for i in range(len(d))
        ]
        # título arriba para el panel central
        axC.set_title("Distancia y PRx por receptor", y=0.98)
        # texto anclado ABAJO al centro
        axC.text(0.5, 0.02, "\n".join(lines), ha="center", va="bottom",
                transform=axC.transAxes, family="monospace", fontsize=10)

        # --- Panel derecho: Real (Sionna RT) ---
        scR = axR.scatter(rx[:,0], rx[:,1], c=prx_right, s=80, cmap="viridis", vmin=vmin, vmax=vmax)
        axR.scatter([drone_xy[0]], [drone_xy[1]], marker="^", s=150, edgecolors="k", facecolors="none", label="Dron")
        for i, (x, y, _) in enumerate(rx):
            label = f"Rx{i}" if not show_values_in_labels else f"Rx{i}\n{prx_right[i]:.1f} dBm"
            axR.text(x + 1, y + 1, label, fontsize=8, weight="bold")
        axR.set_aspect("equal", adjustable="box")
        axR.set_title(title_right)
        # Banner RF debajo del título
        axR.text(0.5, 1.3, rf_str, transform=axR.transAxes, ha="center", va="bottom", fontsize=9)
        axR.set_xlabel("x [m]"); axR.set_ylabel("y [m]")
        axR.grid(True, alpha=0.3)
        fig.colorbar(scR, ax=axR, label="dBm")
        axR.legend(loc="upper right")

        # === Guardado automático ===
        # Carpeta con el nombre que pediste
        out_dir = Path("Environment drones/comparacion PRx teorico real")
        out_dir.mkdir(parents=True, exist_ok=True)

        # Nombre termina en _<frecuencia>
        freq_suffix = f"{fc_ghz:.3f}GHz" if not np.isnan(fc_ghz) else "NA"

        filename = f"comparacion_prx_teo_real_{freq_suffix}.png"

        fig.savefig(out_dir / filename, dpi=150, bbox_inches="tight")

        plt.show()


