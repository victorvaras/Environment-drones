# -*- coding: utf-8 -*-
from __future__ import annotations
import datetime
import socialforce
import torch
from socialforce import potentials
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import sys
from pathlib import Path

# (Se mantiene para no romper ejecuciones fuera del paquete)
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Proyecto
from .sionnaEnv import SionnaRT
from .dron import Dron
from .receptores import ReceptoresManager, Receptor


class DroneEnv(gym.Env):
    """Entorno Gymnasium con Sionna RT (sin imagen de fondo)."""
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 15}

    def __init__(
            self,
            rx_positions: list[tuple[float, float, float]] | None = None,
            rx_goals: list[tuple[float, float, float]] | None = None,
            frequency_mhz: float = 3500.0,
            tx_power_dbm: float = 30.0,
            bandwidth_hz: float = 20e6,
            scene_name: str = "munich",
            antenna_mode: str = "ISO",
            max_steps: int = 400,
            render_mode: str | None = None,
            drone_start: tuple[float, float, float] = (0.0, 0.0, 20.0),

    ):
        super().__init__()
        assert render_mode in (None, "human", "rgb_array"), \
            "render_mode debe ser None, 'human' o 'rgb_array'"
        self.render_mode = render_mode

        self._start = drone_start
        self.max_steps = int(max_steps)
        self.step_count = 0

        # --- Iniciando receptores ---
        self.receptores = ReceptoresManager([Receptor(*p) for p in rx_positions])

        # --- Guardar número de receptores ---
        numero_receptores = self.receptores.n
        rx_velocities_mps = [(0.0, 0.0, 0.0) for _ in range(numero_receptores)]

        #Configuración Social Force Model (SFM) - (Dinámica Peatonal)
        #Se define el "Paso de tiempo" de la física (0.1 s = 10 actualizaciones por segundo)
        dt = 0.1

        #Definición del Potencial de Interacción (Agente-Agente)
        #Esto es lo que controla cómo se evitan los peatones entre sí.
        #-v0: Magnitud de la fuerza de repulsión.
        #-sigma: Rango o alcance de la fuerza (qué tan lejos se "sienten").
        ped_ped_potential = potentials.PedPedPotential2D(
            v0=5.0,  #Fuerza de repulsión
            sigma=1.0,  #Alcance de la fuerza
            asymmetry=0.3  #Factor para "preferir" un lado (esquive)
        )
        ped_ped_potential.delta_t_step = dt #Se sincroniza el 'dt' para el potencial

        #Inicialización del Simulador
        #-oversampling = 1': Sincroniza la física del SFM 1:1 con el paso de Gym.
        #-field_of_view = -1': Permite que el Campo de visión sea de 360°.
        self.sfm_sim = socialforce.Simulator(
            delta_t=dt,
            ped_ped=ped_ped_potential,
            oversampling=1,
            field_of_view =-1
        )

        #Construcción del "Tensor de Estado" (State Tensor)
        #El SFM de la librería socialforce requiere una matriz específica de 10 columnas por agente.
        #Formato: [x, y, vx, vy, ax, ay, gx, gy, tau, v_pref]
        np_initial_states = np.array(rx_positions) #Posiciones iniciales
        np_goal_states = np.array(rx_goals) #Metas (Goals)

        #Se extraen solo las coordenadas X e Y (2D) porque SFM trabaja asi en el plano
        self.sfm_goals_2d = np_goal_states[:, :2]

        #Variables auxiliares para rellenar la matriz o tensor de estado
        #Tensor de estado = (x, y, vx, vy, ax, ay, gx, gy, tau, v_pref)
        initial_velocities_np = np.zeros((numero_receptores, 2))    #vx, vy (velocidades iniciales)
        initial_accelerations_np = np.zeros((numero_receptores, 2)) #ax, ay (aceleraciones iniciales)
        initial_taus_np = np.full((numero_receptores, 1), 0.5) #tau (tiempo de reacción)
        preferred_speeds_np = np.full((numero_receptores, 1), 1.3) #v_pref (velocidad deseada = 1.3 m/s)

        #Se junta en una sola matriz gigante (NumPy)
        initial_state_np = np.hstack([
            np_initial_states[:, :2],   #Columna 0-1: Posición (x, y)
            initial_velocities_np,      #Columna 2-3: Velocidad (vx, vy)
            initial_accelerations_np,   #Columna 4-5: Aceleración (ax, ay)
            self.sfm_goals_2d,          #Columna 6-7: Meta (Goal) (gx, gy)
            initial_taus_np,            #Columna 8  : Tau
            preferred_speeds_np         #Columna 9  : Velocidad Preferida (v_pref)
        ])

        #Se convierte a Tensor de PyTorch (formato requerido para la librería socialforce)
        self.sfm_current_state = torch.tensor(initial_state_np, dtype=torch.float32)

        # --- Iniciando Sionna RT ---
        self.rt = SionnaRT(
            antenna_mode=antenna_mode,
            frequency_mhz=frequency_mhz,
            # tx_power_dbm=tx_power_dbm,
            # bandwidth_hz=bandwidth_hz,
            scene_name=scene_name,
            num_ut=numero_receptores,
            rx_velocities_mps=rx_velocities_mps,
        )

        # --- Construir escena y colocar receptores ---
        self.rt.build_scene()
        self.rt.attach_receivers(self.receptores.positions_xyz())

         
        bounds_min = self.rt.scene_bounds[0]
        bounds_max = self.rt.scene_bounds[1]
        bounds = ((bounds_min[0], bounds_max[0]), (bounds_min[1], bounds_max[1]), (bounds_min[2], bounds_max[2]))
        self.scene_bounds = bounds

        # Dron / spaces
        self.dron = Dron(start_xyz=self._start, bounds=bounds)
        self.action_space = spaces.Box(low=-5.0, high=5.0, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-1e9, high=1e9, shape=(3 + self.receptores.n,), dtype=np.float32
        )
        self.dron.bounds = bounds

        # Estado de render
        self._fig = None
        self._ax = None
        self._canvas = None
        self._ax_map = None
        self._ax_list = None
        self._ax_table = None
        self._sc_rx = None
        self._sc_drone = None
        self._cbar = None
        self._name_texts = []

        # Acumuladores por-UE para la tabla (id -> dict)
        self._acc = None
        self._last_ue_metrics = None

        # ================= Gym API =================

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.step_count = 0

        self.dron = Dron(start_xyz=self._start, bounds=self.scene_bounds)
        self.rt.move_tx(self.dron.pos)

        obs = np.concatenate([self.dron.pos]).astype(np.float32)
        info = {}

        # ---- estado de render ----
        self._fig = None
        self._ax_map = None
        self._ax_list = None
        self._ax_table_top = None
        self._ax_table_br = None
        self._canvas = None

        self._sc_rx = None
        self._sc_drone = None
        self._cbar = None
        self._name_texts = []

        self._last_ue_metrics = []

        # 1) Número de UEs (receptores)
        try:
            self.num_ut = int(self.receptores.positions_xyz().shape[0])
        except Exception:
            # Fallback si tu contenedor expone otra API
            self.num_ut = int(getattr(self.receptores, "num", getattr(self.receptores, "n", 0)))

       

        return obs, info

    def step(self, action: np.ndarray):
        self.step_count += 1

        # Movimiento del dron
        self.dron.step_delta(action)
        self.rt.move_tx(self.dron.pos)

        #---Movimiento de personas / receptores (Rx's)---
        #Social Force Model (SFM)
        #Se calcula la "intención social", el agente o receptor quiere ir a la meta
        #mientras esquiva a otros agentes (fuerza repulsiva peatonal).
        proposed_state = self.sfm_sim(self.sfm_current_state) #Se obtiene el "estado propuesto" por el SFM
        proposed_state_np = proposed_state.detach().numpy()   #Se convierte la respuesta de Torch a NumPy

        #Se obtienen las posiciones actuales para el sensor
        current_positions_3d = np.array([rx.position for rx in self.rt.rx_list])

        proposed_pos_2d = proposed_state_np[:, 0:2]

        #Se extrae la velocidad deseada por el SFM (Vx, Vy)
        proposed_vel_2d = proposed_state_np[:, 2:4]

        #Sensor Híbrido (Sionna Raycast)
        #Escanea la geometría 3D real y devuelve un vector de fuerza
        #que combina repulsión (atrás) y deslizamiento (lateral).
        obstacle_forces = self.rt.get_obstacle_forces(
            current_positions_3d,
            sensor_radius=3.5,  #Rango de visión (Mira a 3.5 metros)
            num_rays=12,        #Resolución del sensor (12 Rayos, buen balance velocidad/precisión)
            force_factor=12.0   #Magnitud de la reacción
        )

        #Integración de Fuerzas
        #Velocidad Final = Velocidad Social + (Fuerza Obstáculo * dt)
        #Se suma la fuerza del muro u obstáculo a la velocidad del SFM
        dt = 0.1
        proposed_vel_2d[:, 0] += obstacle_forces[:, 0] * dt
        proposed_vel_2d[:, 1] += obstacle_forces[:, 1] * dt

        #Integración de Euler (Cálculo de nueva posición)
        #Posición Nueva = Posición Actual + Velocidad Final * dt
        current_pos_2d = self.sfm_current_state.numpy()[:, 0:2]
        final_proposed_pos_2d = current_pos_2d + proposed_vel_2d * dt

        #Validación física y actualización
        #Se prepara el nuevo array de estado validado y actualización
        new_validated_state_np = self.sfm_current_state.numpy().copy()

        for i, rx in enumerate(self.rt.rx_list):
            #Limpieza de tipos de datos (Data Hygiene)
            #Se convierten los Tensores de DrJit/Torch a float de Python puro
            #con la finalidad de evitar conflictos de memoria en NumPy.
            z_clean = float(current_positions_3d[i][2])
            next_x = float(final_proposed_pos_2d[i, 0])
            next_y = float(final_proposed_pos_2d[i, 1])
            vel_x = float(proposed_vel_2d[i, 0])
            vel_y = float(proposed_vel_2d[i, 1])

            #Se construyen las coordenadas candidatas
            pos_actual_clean = np.array([float(current_positions_3d[i][0]), float(current_positions_3d[i][1]), z_clean])
            pos_propuesta_clean = np.array([next_x, next_y, z_clean])

            #Chequeo de Colisión Dura (Hard Collision)
            #Si a pesar de las fuerzas de repulsión, el agente intenta atravesar un muro u obstáculo
            #el motor de física (Sionna) lo detiene.
            if self.rt.is_move_valid_receptores(pos_actual_clean, pos_propuesta_clean):
                #Movimiento Válido: se actualiza la física (posición) y estado
                rx.position = [next_x, next_y, z_clean]
                rx.velocity = [vel_x, vel_y, 0.0]
                new_validated_state_np[i, 0:2] = [next_x, next_y]
                new_validated_state_np[i, 2:4] = [vel_x, vel_y]
            else:
                #Colisión: Detención de emergencia
                rx.velocity = [0.0, 0.0, 0.0]
                new_validated_state_np[i, 2:4] = [0.0, 0.0]

        #Se guarda el estado actualizado para ser utilizado en el siguiente ciclo
        self.sfm_current_state = torch.tensor(new_validated_state_np, dtype=torch.float32)



        # --- Ejecutar paso SYS y obtener métricas ---
        sys_metrics = self.rt.run_sys_step()

        # --- Recompensa ---
        # reward = float(np.mean(snr))
        reward = 1.0

        # --- Observación ---
        obs = np.concatenate([self.dron.pos]).astype(np.float32)

        # --- Terminación ---
        terminated = False
        truncated = self.step_count >= self.max_steps

        info = {
            # --- métricas por UE del step (ya las tenías) ---
            "ue_metrics": sys_metrics["ue_metrics"],

            # --- TBLER acumulada estilo tutorial ---
            # Vector [num_ut] con la TBLER running a este step (idéntica al notebook de Sionna SYS)
            "tbler_running_per_ue": sys_metrics.get("tbler_running_per_ue"),  # list[float] tamaño num_ut

        }
        
        if self.render_mode is None:
            return obs, reward, terminated, truncated, info
        
        elif self.render_mode == "human":
            self._last_ue_metrics = info["ue_metrics"]  # cache para render
            self._last_tbler_running_per_ue = info.get("tbler_running_per_ue", None)
            self._render_to_figure()
        elif self.render_mode == "rgb_array":
            self._last_ue_metrics = info["ue_metrics"]  # cache para render
            self._last_tbler_running_per_ue = info.get("tbler_running_per_ue", None)
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
        self.render_figsize = getattr(self, "render_figsize", (18, 8.5))  # pulgadas
        self.render_dpi = getattr(self, "render_dpi", 120)  # 12.8*100=1280 px, 7.2*100=720 px

        # Usa constrained layout para que Matplotlib “reserve” espacio para la derecha y la colorbar
        self._fig = plt.figure(
            figsize=self.render_figsize,
            dpi=self.render_dpi,
            layout="constrained"  # equivale a set_constrained_layout(True)
        )

        # === TÍTULO GLOBAL ===
        antenna_mode = getattr(self.rt, "antenna_mode", "N/A")
        freq_mhz = getattr(self.rt, "freq_hz", 0) / 1e6
        tx_power = getattr(self.rt, "tx_power_dbm_total", 0)

        title_text = (
            f"Simulación Dron-Receptores | "
            f"Antena: {antenna_mode} | f = {freq_mhz:.1f} MHz | Potencia total = {tx_power:.1f} dBm | "
            f"Step: 0/{self.max_steps}"
        )

        self._suptitle = self._fig.suptitle(
            title_text,
            fontsize=15,
            fontweight='bold',
            y=0.98
        )

        # Gridspec con espacio arriba
        gs = self._fig.add_gridspec(
            1, 2,
            width_ratios=[1.0, 1.2],
            top=0.94  # deja espacio para el título
        )

        # Gridspec principal
        gs = self._fig.add_gridspec(1, 2, width_ratios=[1.0, 1.2])  # un poco más ancho el panel izquierdo

        # Subgrilla izquierda (mapa + lista)
        gs_left = gs[0, 0].subgridspec(2, 1, height_ratios=[0.72, 0.28])

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
        gs_right = gs[0, 1].subgridspec(3, 1, height_ratios=[0.01, 0.55, 0.35], hspace=0.15)
        self._ax_spaces=self._fig.add_subplot(gs_right[0, 0])
        self._ax_spaces.axis("off")
        self._ax_table_top = self._fig.add_subplot(gs_right[1, 0])
        self._ax_table_br = self._fig.add_subplot(gs_right[2, 0])
        for ax in (self._ax_table_top, self._ax_table_br):
            ax.axis("off")
        self._ax_table_top.set_title("Métricas de canal por receptor")

        # Canvas Agg, común a human y rgb_array
        self._canvas = FigureCanvas(self._fig)
        
        if self.render_mode == "human":
            try:
                self._auto_view_2d(margin_ratio=getattr(self, "view_margin", 0.05))
            except Exception:
                pass

            plt.ion()  # modo interactivo
            plt.show(block=False)  # muestra la ventana SIN bloquear

        if self.render_mode == "rgb_array":
            try:
                self._auto_view_2d(margin_ratio=getattr(self, "view_margin", 0.05))
            except Exception:
                pass

    def _render_common(self):
        import numpy as np
        import matplotlib.pyplot as plt

        self._ensure_figure()

        # --- Datos base ---
        prx = np.asarray(self.rt.compute_prx_dbm(), dtype=float).reshape(-1)
        rx = self.receptores.positions_xyz()  # shape (N, 3)
        drone_xyz = np.asarray(self.dron.pos, dtype=float).reshape(3)

        # === ACTUALIZAR STEP EN EL TÍTULO ===
        if hasattr(self, '_suptitle'):
            antenna_mode = getattr(self.rt, "antenna_mode", "N/A")
            freq_mhz = getattr(self.rt, "freq_hz", 0) / 1e6
            tx_power = getattr(self.rt, "tx_power_dbm_total", 0)

            title_text = (
                f"Simulación Dron-Receptores | "
                f"Antena: {antenna_mode} | f = {freq_mhz:.1f} MHz | Potencia total = {tx_power:.1f} dBm | "
                f"Step: {self.step_count}/{self.max_steps}"
            )

            self._suptitle.set_text(title_text)

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
            self._name_texts.append(self._ax_map.text(drone_xyz[0] + 1.0, drone_xyz[1] + 1.0,
                                                      "Drone", fontsize=9, weight="bold"))
            for i, (x, y, _) in enumerate(rx):
                self._name_texts.append(self._ax_map.text(x + 1.0, y + 1.0, f"Rx{i}", fontsize=8))
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
            self._name_texts[0].set_position((drone_xyz[0] + 1.0, drone_xyz[1] + 1.0))
            for i, (x, y, _) in enumerate(rx):
                self._name_texts[i + 1].set_position((x + 1.0, y + 1.0))

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
        self._ax_table_top.clear()
        self._ax_table_top.axis("off")
        self._ax_table_top.set_title(
            "Effective SINR, Spectral Efficiency, Shannon y Achieved TBLER (step & running)"
        )

        ue_metrics = getattr(self, "_last_ue_metrics", [])
        tbler_running_per_ue = getattr(self, "_last_tbler_running_per_ue", None)

        if not ue_metrics:
            self._ax_table_top.text(0.02, 0.95, "Sin métricas aún (esperando primer step)...",
                                    va="top", ha="left", fontsize=7, family="monospace")
        else:
            # Encabezados (corrigimos columnas y añadimos TBLER_running)
            headers = ["Receptor", "SINR eff(dB)", "SE(b/Hz)", "Shannon(b/Hz)", "SE vs Shannon(%)", "TBLER step",
                       "TBLER running"]
            line = "  ".join(f"{h:>14s}" for h in headers)
            sep = "-" * len(line)
            rows = [line, sep]

            # Mapeo + formateo
            def fmt(x, nd):
                try:
                    xf = float(x)
                    return f"{xf:.{nd}f}" if np.isfinite(xf) else "  NaN"
                except Exception:
                    return "  NaN"

            # Ordenar por id
            for m in sorted(ue_metrics, key=lambda x: x["ue_id"]):
                i = int(m["ue_id"])
                sinr = m.get("sinr_eff_db", float('nan'))
                se_la = m.get("se_la", float('nan'))
                se_sh = m.get("se_shannon", float('nan'))
                gap = m.get("se_gap_pct", float('nan'))
                # OJO: tu dict usa la clave "tbler" (no "bler")
                tbler_step = m.get("tbler", float('nan'))

                # TBLER running (si no está disponible aún, deja NaN)
                tbler_run = float('nan')
                if tbler_running_per_ue is not None and i < len(tbler_running_per_ue):
                    tbler_run = tbler_running_per_ue[i]

                rows.append("  ".join([
                    f"{('Rx' + str(i)):>14s}",
                    f"{fmt(sinr, 2):>14s}",
                    f"{fmt(se_la, 3):>14s}",
                    f"{fmt(se_sh, 3):>14s}",
                    f"{fmt(gap, 1):>14s}",
                    f"{fmt(tbler_step, 3):>14s}",
                    f"{fmt(tbler_run, 3):>14s}",
                ]))

            # Leyenda breve con bler_target
            legend_lines = []
            legend_lines.append("TBLER step: 0 = ACK, 1 = NACK, NaN = no agendado")
            legend_lines.append("TBLER running: 1 - ACK acum / TX acum")

            full_text = "\n".join(rows + ["", *legend_lines])
            self._ax_table_top.text(0.01, 0.98, full_text,
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

    def _auto_view_2d(self, margin_ratio: float = 0.05):
        """
        Ajusta la vista 2D usando los límites (min y max) entregados por SionnaRT.
        Usa self.rt.scene_bounds = (min_xyz, max_xyz)
        margin_ratio: margen porcentual extra alrededor de la escena.
        """
        import numpy as np

        # --- Recuperar límites desde SionnaRT ---
        if hasattr(self.rt, "scene_bounds"):
            mn, mx = self.rt.scene_bounds
        else:
            raise AttributeError("No se encontraron los límites de la escena (scene_bounds) en self.rt")

        mn = np.array(mn, dtype=float)
        mx = np.array(mx, dtype=float)

        # --- Tomar solo las coordenadas X e Y ---
        xmin, xmax = mn[0], mx[0]
        ymin, ymax = mn[1], mx[1]

        # --- Calcular tamaño y margen ---
        w = max(1e-6, xmax - xmin)
        h = max(1e-6, ymax - ymin)
        mxr = w * margin_ratio
        myr = h * margin_ratio

        # --- Aplicar los límites a los ejes ---
        self._ax_map.set_aspect("equal", adjustable="box")
        self._ax_map.set_xlim(xmin - mxr, xmax + mxr)
        self._ax_map.set_ylim(ymin - myr, ymax + myr)
        self._ax_map.grid(True, alpha=0.3)

        # --- Evitar autoescala ---
        self._ax_map.autoscale(enable=False)
        self._ax_map.autoscale_view(tight=True)

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

    """
    # ================= Render avanzado (dual snapshot) =================
    # (puedes llamarlo desde fuera del entorno, pasando PRx teórico y real)

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

    """