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
from .sionnaEnv import SionnaRT, _goodput_routeA_bits
from .dron import Dron
from .receptores import ReceptoresManager, Receptor


class DroneEnv(gym.Env):
    """Entorno Gymnasium con Sionna RT (sin imagen de fondo)."""
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 15}

    def _get_num_ut(self) -> int:
        """Devuelve el número de receptores (UT) de forma robusta."""
        try:
            # Si tu contenedor de receptores expone un contador
            return int(getattr(self.receptores, "num", getattr(self.receptores, "n")))
        except Exception:
            # Fallback: contar posiciones
            return int(self.receptores.positions_xyz().shape[0])

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

        step_duration_seconds: float = 0.01,   # <-- NUEVO (10 ms por defecto)
        scs_khz: float = 30.0,                 # <-- NUEVO (SCS típica)
        n_prb: int = 16,                      # <-- NUEVO (tu valor)
        zeta: float = 0.85,                   # NUEVO
        overhead: float = 0.20,                # NUEVO
        bler_target: float = 0.10,              # NUEVO
        tb_size_bits: int = 10000,   # <-- NUEVO: tamaño de bloque virtual (bits)
        
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
        self._ax_map = None
        self._ax_list = None
        self._ax_table = None

        self._drone_sc = None
        self._sc = None
        self._texts = []
        self._cbar = None

        self._sc_rx = None
        self._sc_drone = None
        self._cbar = None
        self._name_texts = []

        # Acumuladores por-UE para la tabla (id -> dict)
        self._acc = None  # se creará en reset()
        self._last_ue_metrics = None  # ya lo usabas

        # Parámetros de enlace, calculo goodput
        self.step_duration_seconds = float(step_duration_seconds)
        self.scs_khz = float(scs_khz)
        self.n_prb = int(n_prb)
        self._zeta = float(zeta)
        self._overhead = float(overhead)
        self._bler_t = float(bler_target)
        self._last_ue_metrics = None          # <-- NUEVO: cache de métricas para render
        self._ax_gp = None         # <-- NUEVO: eje para goodput
        self._bars_gp = None       # <-- NUEVO: barras de goodput
        self.tb_size_bits = int(tb_size_bits)   # <-- NUEVO



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

        # ---- estado de render ----
        self._fig = None
        self._ax_map = None
        self._ax_list = None
        self._ax_table_top = None 
        self._ax_table_br  = None  
        self._canvas = None

        self._sc_rx = None
        self._sc_drone = None
        self._cbar = None
        self._name_texts = []

        self._last_ue_metrics = []
        self._last_blocks_summary = None


        # 1) Número de UEs (receptores)
        try:
            self.num_ut = int(self.receptores.positions_xyz().shape[0])
        except Exception:
            # Fallback si tu contenedor expone otra API
            self.num_ut = int(getattr(self.receptores, "num", getattr(self.receptores, "n", 0)))

        # 2) Acumuladores por-UE para la tabla superior (intent/os/éxitos por UE)
        self._acc = {i: {"bits_ok": 0, "attempts": 0, "successes": 0} for i in range(self.num_ut)}

        # 3) Acumuladores por-UE para la tabla inferior derecha (bloques y bits)
        self.blocks_acc_tx = [0 for _ in range(self.num_ut)]   # intentos (TX) acumulados
        self.blocks_acc_ok = [0 for _ in range(self.num_ut)]   # éxitos (OK) acumulados
        self.bits_acc_total = [0 for _ in range(self.num_ut)]  # bits OK acumulados

        # (Si usabas un dict de bloques, mantenlo alineado con num_ut)
        self.blocks_acc = {i: {"tx": 0, "ok": 0} for i in range(self.num_ut)}

        # 4) Estado para el render
        self._last_ue_metrics = []
        self._last_blocks_summary = None



        return obs, info

    def step(self, action: np.ndarray):
        self.step_count += 1

        # Movimiento del dron
        self.dron.step_delta(action)
        self.rt.move_tx(self.dron.pos)

        #Movimiento de personas

        # Señal / SNR

        
        
        #prx = self.rt.compute_prx_dbm()
        #snr = self.rt.compute_snr_db(prx)
        snr = 1.0
        prx = None
        reward = float(np.mean(snr))


        # Ejecutar paso SYS y obtener métricas
        sys_metrics = self.rt.run_sys_step()

        # reward = float(tf.reduce_mean(sys_metrics["sinr_eff_db_true"]).numpy())

        
        terminated = False
        truncated = self.step_count >= self.max_steps

        #obs = np.concatenate([self.dron.pos, prx]).astype(np.float32)
        obs = np.concatenate([self.dron.pos]).astype(np.float32)

        info = {"ue_metrics": sys_metrics["ue_metrics"],
                "pf_metric": sys_metrics.get("pf_metric"),
                "ut_scheduled": sys_metrics.get("ut_scheduled"),
                "step_blocks_summary": sys_metrics.get("step_blocks_summary")} 
        
        self._last_ue_metrics = info["ue_metrics"]  # cache para render
        self._last_blocks_summary = info["step_blocks_summary"]

        if self.render_mode == "human":
            self._render_to_figure()
        elif self.render_mode == "rgb_array":
            frame = self._render_to_array()
            info["frame"] = frame

        return obs, reward, terminated, truncated, info

    # ================= Render helpers =================
    def _ensure_figure(self):
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

        if self._fig is not None and self._ax_map is not None:
            return

        # ---- define resolución exacta (cambia si quieres 1920x1080) ----
        self.render_figsize = getattr(self, "render_figsize", (16, 7.5))  # pulgadas
        self.render_dpi     = getattr(self, "render_dpi", 110)              # 12.8*100=1280 px, 7.2*100=720 px

        # Usa constrained layout para que Matplotlib “reserve” espacio para la derecha y la colorbar
        self._fig = plt.figure(
            figsize=self.render_figsize,
            dpi=self.render_dpi,
            layout="constrained"  # equivale a set_constrained_layout(True)
        )

        # Gridspec principal
        gs = self._fig.add_gridspec(1, 2, width_ratios=[1.45, 1.20])  # un poco más ancho el panel izquierdo

        # Subgrilla izquierda (mapa + lista)
        gs_left = gs[0, 0].subgridspec(2, 1, height_ratios=[0.68, 0.32])

        self._ax_map = self._fig.add_subplot(gs_left[0, 0])
        self._ax_map.set_aspect("equal", adjustable="box")
        self._ax_map.set_title("Vista 2D: Dron y Receptores")
        self._ax_map.set_xlabel("x [m]")
        self._ax_map.set_ylabel("y [m]")
        self._ax_map.grid(True, alpha=0.3)

        self._ax_list = self._fig.add_subplot(gs_left[1, 0])
        self._ax_list.set_title("Posiciones y PRx (dBm)")
        self._ax_list.axis("off")

        # Subgrilla derecha (tablas: arriba métricas, abajo bloques)
        gs_right = gs[0, 1].subgridspec(2, 1, height_ratios=[0.55, 0.45])
        self._ax_table_top = self._fig.add_subplot(gs_right[0, 0])
        self._ax_table_br  = self._fig.add_subplot(gs_right[1, 0])
        for ax in (self._ax_table_top, self._ax_table_br):
            ax.axis("off")
        self._ax_table_top.set_title("Métricas de canal por receptor")
        self._ax_table_br.set_title("Bloques: éxito por step y acumulado")

        # Canvas Agg, común a human y rgb_array
        self._canvas = FigureCanvas(self._fig)

        # No uses tight_layout junto con constrained; si lo tenías, elimínalo.
        if self.render_mode == "human":
            plt.ion()
            plt.show(block=False)


    def _render_common(self):
        import numpy as np
        import matplotlib.pyplot as plt

        self._ensure_figure()

        # --- Datos base ---
        prx = np.asarray(self.rt.compute_prx_dbm(), dtype=float).reshape(-1)
        rx = self.receptores.positions_xyz()  # shape (N, 3)
        drone_xyz = np.asarray(self.dron.pos, dtype=float).reshape(3)

        # ===== MAPA (izq/arriba) =====
        if self._sc_rx is None:
            # Dron
            self._sc_drone = self._ax_map.scatter([drone_xyz[0]], [drone_xyz[1]],
                                                s=140, marker="^", zorder=3, label="Drone")
            # Receptores coloreados por PRx
            self._sc_rx = self._ax_map.scatter(rx[:, 0], rx[:, 1], s=90, c=prx,
                                            cmap="viridis", zorder=2)
            # Etiquetas con nombres (Drone, Rx0, Rx1, …)
            # Nota: mostramos nombre al lado del punto
            self._name_texts = []
            self._name_texts.append(self._ax_map.text(drone_xyz[0]+1.0, drone_xyz[1]+1.0,
                                                    "Drone", fontsize=9, weight="bold"))
            for i, (x, y, _) in enumerate(rx):
                self._name_texts.append(self._ax_map.text(x+1.0, y+1.0, f"Rx{i}", fontsize=8))
            # Colorbar
            if self._cbar is None:
                self._cbar = self._fig.colorbar(
                    self._sc_rx, ax=self._ax_map, label="PRx [dBm]",
                    fraction=0.046, pad=0.04  # más compacta y con espacio
                )
            else:
                self._cbar.update_normal(self._sc_rx)
        else:
            # actualizar posiciones/colores
            self._sc_drone.set_offsets([[drone_xyz[0], drone_xyz[1]]])
            self._sc_rx.set_offsets(rx[:, :2])
            self._sc_rx.set_array(prx)
            # actualizar textos (posiciones)
            self._name_texts[0].set_position((drone_xyz[0]+1.0, drone_xyz[1]+1.0))
            for i, (x, y, _) in enumerate(rx):
                self._name_texts[i+1].set_position((x+1.0, y+1.0))

        # ===== LISTA (izq/abajo): posiciones + PRx =====
        # Construimos un texto monoespaciado
        lines = []
        lines.append("ID      x[m]      y[m]      z[m]      PRx[dBm]")
        lines.append("------------------------------------------------")
        lines.append(f"{'Drone':6s}  {drone_xyz[0]:7.2f}  {drone_xyz[1]:7.2f}  {drone_xyz[2]:7.2f}      -")
        for i, (x, y, z) in enumerate(rx):
            prx_i = float(prx[i])
            lines.append(f"Rx{i:02d}   {x:7.2f}  {y:7.2f}  {z:7.2f}   {prx_i:10.2f}")
        text_block = "\n".join(lines)

        self._ax_list.clear()
        self._ax_list.set_title("Posiciones y PRx (dBm)")
        self._ax_list.axis("off")
        self._ax_list.text(0.01, 0.98, text_block, va="top", ha="left",
                        family="monospace", fontsize=9)


        # ===== TABLA DERECHA SUPERIOR: métricas por-UE =====
        ue_metrics = getattr(self, "_last_ue_metrics", [])

        self._ax_table_top.clear()
        self._ax_table_top.axis("off")
        self._ax_table_top.set_title("Métricas de canal por receptor")

        if not ue_metrics:
            self._ax_table_top.text(0.02, 0.95, "Sin métricas aún (esperando primer step)...",
                                    va="top", ha="left", fontsize=9, family="monospace")
        else:
            # Construye tabla por-UE
            headers = ["Receptor", "Bits OK (step)", "Bits OK (acum.)",
                    "Intentos", "Éxitos", "Tasa éxito", "MCS", "SINR(dB)"]
            line = "  ".join(f"{h:>14s}" for h in headers)
            sep = "-" * len(line)
            rows = [line, sep]

            # Si guardas tú los acumulados por-UE en self._acc (arriba-izquierda), úsalos. Si no, puedes
            # mostrar solo lo del step y MCS/SINR.
            # Aquí uso los campos que añadimos en ue_metrics (blocks_* y success_rate_*), si existen.
            for m in sorted(ue_metrics, key=lambda x: x["ue_id"]):
                i   = int(m["ue_id"])
                bok = int(m.get("num_decoded_bits", 0))
                sinr= float(m.get("sinr_eff_db", 0.0))
                mcs = int(m.get("mcs_index", 0))

                # Si vienen los acumulados del run_sys_step:
                txA = m.get("blocks_tx_accum", 0)
                okA = m.get("blocks_ok_accum", 0)
                rateA = float(m.get("success_rate_accum", 0.0))

                # Y por-step:
                txS = m.get("blocks_tx_step", 0)
                okS = m.get("blocks_ok_step", 0)
                rateS = float(m.get("success_rate_step", 0.0))

                rows.append("  ".join([
                    f"{('Rx'+str(i)):>14s}",
                    f"{okS:>14d}",
                    f"{int(m.get('bits_ok_accum', 0)):>14d}",
                    f"{txA:>14d}",
                    f"{okA:>14d}",
                    f"{(rateA*100.0):>13.1f}%",
                    f"{mcs:>14d}",
                    f"{sinr:>14.1f}",
                ]))

            self._ax_table_top.text(0.01, 0.98, "\n".join(rows),
                                    va="top", ha="left", family="monospace", fontsize=9)

        # Actualización de la colorbar
        if self._cbar is not None:
            self._cbar.update_normal(self._sc_rx)


        # ===== TABLA DERECHA INFERIOR: BLOQUES =====
        self._ax_table_br.clear()
        self._ax_table_br.axis("off")
        self._ax_table_br.set_title("Bloques: éxito por step y acumulado")

        bs = getattr(self, "_last_blocks_summary", None)
        if not bs:
            self._ax_table_br.text(0.02, 0.95, "Sin datos de bloques aún...",
                                va="top", ha="left", fontsize=9, family="monospace")
        else:
            headers = ["Receptor", "TX(step)", "OK(step)", "%OK(step)", "TX(acum)", "OK(acum)", "%OK(acum)"]
            line = "  ".join(f"{h:>10s}" for h in headers)
            sep  = "-" * len(line)
            rows = [line, sep]

            n = len(bs.get("blocks_tx_step_per_ue", []))
            for i in range(n):
                tx_s = int(bs["blocks_tx_step_per_ue"][i])
                ok_s = int(bs["blocks_ok_step_per_ue"][i])
                rs_s = float(bs["success_rate_step_per_ue"][i]) * 100.0

                tx_a = int(bs["blocks_tx_accum_per_ue"][i])
                ok_a = int(bs["blocks_ok_accum_per_ue"][i])
                rs_a = float(bs["success_rate_accum_per_ue"][i]) * 100.0

                rows.append("  ".join([
                    f"{('Rx'+str(i)):>10s}",
                    f"{tx_s:>10d}", f"{ok_s:>10d}", f"{rs_s:>10.1f}",
                    f"{tx_a:>10d}", f"{ok_a:>10d}", f"{rs_a:>10.1f}",
                ]))

            rows.append(sep)
            tot_tx = int(bs["blocks_tx_step_total"])
            tot_ok = int(bs["blocks_ok_step_total"])
            tot_rs = float(bs["success_rate_step_total"]) * 100.0
            rows.append(f"{'TOTAL (step)':>10s}  {tot_tx:>10d}  {tot_ok:>10d}  {tot_rs:>10.1f}")

            self._ax_table_br.text(0.01, 0.98, "\n".join(rows),
                                va="top", ha="left", family="monospace", fontsize=9)



    def _render_to_figure(self):
        import matplotlib.pyplot as plt
        self._ensure_figure()
        self._render_common()
        self._fig.canvas.draw()
        self._fig.canvas.flush_events()
        plt.pause(1.0 / max(1, self.metadata.get("render_fps", 5)))

    def _render_to_array(self) -> np.ndarray:
        """
        Dibuja la figura en el canvas Agg y devuelve un frame RGB (H, W, 3) uint8.
        """
        import numpy as np

        # Asegura figura + ejes y pinta el contenido
        self._ensure_figure()
        self._render_common()

        # Dibuja en el canvas Agg
        self._fig.canvas.draw()

        # Tamaño en píxeles
        w, h = self._fig.canvas.get_width_height()
        # Buffer RGBA (bytes) -> ndarray (h, w, 4)
        buf = self._canvas.buffer_rgba()
        rgba = np.frombuffer(buf, dtype=np.uint8).reshape((h, w, 4))

        # Quita alpha y copia (para que no sea una vista de solo-lectura)
        rgb = rgba[:, :, :3].copy()
        return rgb


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
        self._ax_gp = None
        self._bars_gp = None
        self._bar_labels = []



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


