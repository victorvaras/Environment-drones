#Importaciones
from __future__ import annotations
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

#LLamados del Proyecto
from .sionnaEnv import SionnaRT
from env.environment.droneVelocityEnv import DroneVelocityEnv, DroneVelocityEnvConfig
from .receptores_mobility import ReceptorMobilityManager

class DroneEnv(gym.Env):
    """
    Entorno personalizado de Gymnasium para optimización de redes UAV mediante RL.

    Integra simulaciones electromagnéticas de alta fidelidad (Sionna Ray Tracing),
    movilidad peatonal autónoma (Social Force Model) y dinámica de vuelo de drones.
    El agente RL (Dron) debe aprender a navegar el espacio aéreo maximizando
    la calidad de servicio de los usuarios terrestres.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 15}

    def __init__(
            self,
            step_durations: float = 0.1,                                    #Resolución temporal de la simulación
            rx_positions: list[tuple[float, float, float]] | None = None,   #Posiciones iniciales de los receptores
            rx_goals: list[tuple[float, float, float]] | None = None,       #Metas de los receptores
            num_agents: int = 10,                                           #Número de receptores
            frequency_mhz: float = 3500.0,
            tx_power_dbm: float = 30.0,
            bandwidth_hz: float = 20e6,
            scene_name: str = "munich",
            antenna_mode: str = "ISO",
            max_steps: int = 400,                                           #Numero de steps para finalizar la simulación
            render_mode: str | None = None,
            drone_start: tuple[float, float, float] = (0.0, 0.0, 20.0),
            run_metrics: bool = False,                                      #Si es False = simulación rápida, True = simulación completa
            mode_set_vuelo: int = 7,                                        #Modo de vuelo del dron para la simulación
    ):
        super().__init__()
        assert render_mode in (None, "human", "rgb_array"), \
            "render_mode debe ser None, 'human' o 'rgb_array'"
        self.render_mode = render_mode

        self._start = drone_start
        self.sim_dt = step_durations
        self.max_steps = int(max_steps)
        self.step_count = 0
        self.run_metrics = run_metrics

        #1.Configuración de la Cantidad de receptores
        #Si se le asignan posiciones de manera estática
        if rx_positions is not None:
            self.current_num_agents = len(rx_positions)
            using_manual_spawn = True #Se utilizara de manera manual
        else:
            self.current_num_agents = num_agents
            using_manual_spawn = False #Se utilizara el SpawnManager

        #Se guardan las referencias manuales para el reset de la simulación
        self._manual_rx_pos = rx_positions if using_manual_spawn else None
        self._manual_rx_goals = rx_goals if using_manual_spawn else None

        #Se inicializan las Velocidades iniciales para el cálculo de Doppler
        rx_velocities_mps = [(0.0, 0.0, 0.0) for _ in range(self.current_num_agents)]

        #2.Sionna RT
        self.rt = SionnaRT(
            antenna_mode=antenna_mode,
            frequency_mhz=frequency_mhz,
            # tx_power_dbm=tx_power_dbm,
            # bandwidth_hz=bandwidth_hz,
            scene_name=scene_name,
            num_ut=self.current_num_agents,      #Se reserva memoria para N receptores
            rx_velocities_mps=rx_velocities_mps,
        )
        self.rt.build_scene() #Se construye la escena en Sionna

        #3.Gestor de movilidad (Física y Navegación)
        #Se inicializa el Manager pasandole los parámetros físicos
        self.mobility_manager = ReceptorMobilityManager(
            sionna_rt=self.rt,                                                   #Sionna
            bounds_min=(self.rt.scene_bounds[0][0], self.rt.scene_bounds[0][1]), #Limites minimos de la escena
            bounds_max=(self.rt.scene_bounds[1][0], self.rt.scene_bounds[1][1]), #Limites máximos de la escena
            dt_sim=self.sim_dt,                                                  #Paso de tiempo (dt de tiempo)
            sfm_v0=5.0, sfm_sigma=0.5, sfm_u0=80.0, sfm_r=0.5                    #Parametros SFM
        )

        #4.Extracción de Obstáculos (Slicer)
        #Se extrae la geometría estática de Sionna una sola vez.
        #Se utiliza la función 'get_sfm_obstacles'.
        print(f"[Gym] Extrayendo obstáculos de la escena '{scene_name}'...")

        #Lógica de Auto-Escalado (Auto-Scale) para densidad del scanner
        #Se calcula el tamaño del mapa para decidir la densidad y proteger la RAM.
        bounds = self.rt.mi_scene.bbox()
        extent = max(bounds.max.x - bounds.min.x, bounds.max.y - bounds.min.y)

        if extent > 1000.0:
            gym_density = 1.5  #Escena Grande -> Menos resolución
            print(f"[Gym] Escena Gigante ({extent:.0f}m). Ajustando densidad a: {gym_density}m")
        elif extent > 500.0:
            gym_density = 0.8  #Escena mediana
            print(f"[Gym] Escena mediana ({extent:.0f}m). Ajustando densidad a: {gym_density}m")
        else:
            gym_density = 0.4  #Escena pequeña -> Alta precisión
            print(f"[Gym] Escena pequeña o estándar ({extent:.0f}m). Usando alta precisión: {gym_density}m")

        #Se utiliza el escaner para obtener los obstáculos para la API Socialforce (Slicer)
        #grid_density = densidad calculada dinámicamente
        obstacles_np = self.rt.get_sfm_obstacles(grid_density=gym_density)

        #Se configura el manager con los obstáculos
        self.mobility_manager.configure_obstacles(obstacles_np)

        #Se inicializa self.receptores como None (se le asigna valor en reset)
        self.receptores = None

        #5.Configuración Final del Entorno (Bounds, Spaces, Rendering)
        #Bounds y Dron
        #Se definen los límites del espacio de acción y observación para RL
        scene_bounds = ((self.rt.scene_bounds[0][0], self.rt.scene_bounds[1][0]),
                        (self.rt.scene_bounds[0][1], self.rt.scene_bounds[1][1]),
                        (self.rt.scene_bounds[0][2], self.rt.scene_bounds[1][2]))
        self.scene_bounds = scene_bounds

        #Espacios de Gymnasium (Espacios de Acción y Observación)
        self.action_space = spaces.Box(low=-5.0, high=5.0, shape=(3,), dtype=np.float32)

        #Se asume N fijo para el shape del espacio de observación
        self.observation_space = spaces.Box(
            low=-1e9, high=1e9, shape=(3 + self.current_num_agents,), dtype=np.float32
        )

        #Inicializacion para movimiento de dron realista
        self.mode_set_vuelo = mode_set_vuelo  

        cfg = DroneVelocityEnvConfig(
            start_xyz=self._start,
            start_rpy=(0.0, 0.0, 0.0),
            control_hz=120,
            physics_hz=240,
            mode=self.mode_set_vuelo,
            render=False,
            drone_model="cf2x",
            seed=42,
            record_trajectory=True,
        )

        self.dron_Realista = DroneVelocityEnv(
            cfg = cfg,
            step_durations = self.sim_dt)

        #Variables de Renderizado
        self._init_render_vars()

    # ================= Gym API =================
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """
        Reinicia el entorno al estado inicial (t=0).
        Purga las memorias físicas de Sionna, reubica al Dron y lanza un nuevo
        ciclo de generación de receptores (UEs).
        """
        super().reset(seed=seed)
        self.step_count = 0

        #Sincronización con Sionna
        #Se mueve el transmisor a la posición inicial para que el cálculo sea correcto desde t=0
        self.dron_Realista.reset()
        self.rt.move_tx(self._start, (0.0, 0.0, 0.0))

        #Se realiza la limpieza de los receptores previos de Sionna antes de crear nuevos
        self.rt.remove_receivers()

        #Reinicio de receptores (Manager)
        #El Manager se encarga de: Spawn, Metas, SFM Reset
        self.receptores = self.mobility_manager.reset(
            num_agents=self.current_num_agents,  #Número de receptores
            rx_positions=self._manual_rx_pos,    #Posiciones iniciales
            rx_goals=self._manual_rx_goals,      #Metas
            seed=seed                            #Semilla
        )

        #Sincronización con Sionna (attach_receivers)
        #El manager crea los objetos, pero el entorno los conecta al RT.
        self.rt.attach_receivers(self.mobility_manager.get_positions_xyz())

        #Se expone el simulador para visualización externa
        self.sfm_sim = self.mobility_manager.sfm_sim

        #Regenerar primera observación (Observación inicial)
        obs = np.concatenate([self._start]).astype(np.float32)
        info = {}

        #Limpieza de estado de renderizado y métricas
        #Se borran las referencias gráficas para evitar superposiciones en nuevos episodios
        self._init_render_vars()
        self._last_ue_metrics = []
        self.num_ut = len(self.receptores)
        self.dron_Realista.reset()

        return obs, info

    def step(self, action: np.ndarray):
        """
        Ejecuta un paso o ciclo de la simulación.

        1. Realiza el vuelo del dron (dron_Realista).
        2. Utiliza la dinámica peatonal (SFM).
        """
        self.step_count += 1

        #1.Movimiento del Dron
        movimiento_normalizado = self.dron_Realista.step_move(action, dt=self.sim_dt)
        movimiento_valido = self.rt.is_move_valid(self.rt.tx.position, movimiento_normalizado )
        drone_velocity_mps = self.dron_Realista.get_velocity()
          
        self.rt.move_tx(movimiento_normalizado, drone_velocity_mps)       

        #2.Movimiento de Receptores
        #SFM + Control Reactivo + Doppler + Validación
        self.mobility_manager.step()

        #3.Métricas y Sionna SYS
        info = self._get_metrics_info()
        #sys_metrics = self.rt.run_sys_step()

        #Recompensa
        reward = 1.0

        #Observación
        obs = np.concatenate([movimiento_normalizado]).astype(np.float32)

        # --- Terminación ---
        #movimiento_valido = True
        if (movimiento_valido):
            terminated = False
        else:
            terminated = True

        truncated = self.step_count >= self.max_steps

        #Renderizado
        if self.render_mode is not None:
            self._handle_render(info)

        return obs, reward, terminated, truncated, info

    # ================= Render helpers (Visualización y UI) =================
    def _ensure_figure(self):
        """
        Inicializa la figura de Matplotlib estructurada con GridSpec.

        Se ejecuta una sola vez. Configura los lienzos para el mapa 2D,
        la lista de posiciones y la tabla de métricas de telecomunicaciones.
        """
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

        if self._fig is not None and self._ax_map is not None:
            return

        #Resolución optimizada para grabación de video
        self.render_figsize = getattr(self, "render_figsize", (18, 8.5))  #Pulgadas
        self.render_dpi = getattr(self, "render_dpi", 120)

        #constrained_layout evita superposición de ejes y barras de color
        self._fig = plt.figure(
            figsize=self.render_figsize,
            dpi=self.render_dpi,
            layout="constrained" #Equivale a set_constrained_layout(True)
        )

        # --- Encabezado Global ---
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

        # --- Maquetación de la interfaz (GridSpec) ---
        gs = self._fig.add_gridspec(
            1, 2,
            width_ratios=[1.0, 1.2],
            top=0.94
        )

        #Gridspec principal
        gs = self._fig.add_gridspec(1, 2, width_ratios=[1.0, 1.2])

        #Panel Izquierdo: Mapa 2D y log de posiciones
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

        #Panel Derecho: Tablas de métricas
        gs_right = gs[0, 1].subgridspec(3, 1, height_ratios=[0.01, 0.55, 0.35], hspace=0.15)
        self._ax_spaces=self._fig.add_subplot(gs_right[0, 0])
        self._ax_spaces.axis("off")
        self._ax_table_top = self._fig.add_subplot(gs_right[1, 0])
        self._ax_table_br = self._fig.add_subplot(gs_right[2, 0])
        for ax in (self._ax_table_top, self._ax_table_br):
            ax.axis("off")
        self._ax_table_top.set_title("Métricas de canal por receptor")

        #Canvas Agg: Permite renderizar a un array de NumPy sin necesidad de interfaz gráfica (Headless mode)
        self._canvas = FigureCanvas(self._fig)

        #Modo interactivo para visualización humana en tiempo real
        if self.render_mode == "human":
            try:
                self._auto_view_2d(margin_ratio=getattr(self, "view_margin", 0.05))
            except Exception:
                pass
            plt.ion()
            plt.show(block=False)

        if self.render_mode == "rgb_array":
            try:
                self._auto_view_2d(margin_ratio=getattr(self, "view_margin", 0.05))
            except Exception:
                pass

    def _render_common(self):
        """
        Actualiza dinámicamente los datos del UI sin reconstruir la figura completa.

        Por eficiencia computacional durante el entrenamiento RL, utiliza `set_offsets`
        para mover los puntos en el scatter plot, evitando la costosa operación de
        limpiar y redibujar (ax.clear()) los ejes en cada step.
        """
        import numpy as np
        import matplotlib.pyplot as plt

        self._ensure_figure()

        # --- Extracción de estado actual ---
        prx = np.asarray(self.rt.compute_prx_dbm(), dtype=float).reshape(-1)
        rx = self.receptores.positions_xyz()  # shape (N, 3)
        drone_xyz = np.asarray(self._start, dtype=float).reshape(3)

        # --- Actualización de Título ---
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

        # --- Actualización del Mapa 2D ---
        if self._sc_rx is None:
            #Dron
            self._sc_drone = self._ax_map.scatter([drone_xyz[0]], [drone_xyz[1]],
                                                  s=140, marker="^", zorder=3, label="Drone")
            #Receptores
            self._sc_rx = self._ax_map.scatter(rx[:, 0], rx[:, 1], s=90, c=prx,
                                               cmap="viridis", zorder=2)
            #Etiquetas con nombres (Drone, Rx0, Rx1, …)
            self._name_texts = []
            self._name_texts.append(self._ax_map.text(drone_xyz[0] + 1.0, drone_xyz[1] + 1.0,
                                                      "Drone", fontsize=9, weight="bold"))
            for i, (x, y, _) in enumerate(rx):
                self._name_texts.append(self._ax_map.text(x + 1.0, y + 1.0, f"Rx{i}", fontsize=8))

            #Colorbar
            if self._cbar is None:
                self._cbar = self._fig.colorbar(
                    self._sc_rx, ax=self._ax_map, label="PRx [dBm]",
                    fraction=0.046, pad=0.04
                )
            else:
                self._cbar.update_normal(self._sc_rx)
        else:
            #Actualización eficiente (posiciones y colores)
            self._sc_drone.set_offsets([[drone_xyz[0], drone_xyz[1]]])
            self._sc_rx.set_offsets(rx[:, :2])
            self._sc_rx.set_array(prx)

            self._name_texts[0].set_position((drone_xyz[0] + 1.0, drone_xyz[1] + 1.0))
            for i, (x, y, _) in enumerate(rx):
                self._name_texts[i + 1].set_position((x + 1.0, y + 1.0))

        # --- Actualización Panel Izquierdo Inferior ---
        #Construimos un texto monoespaciado
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

        # --- Actualización Panel Derecho: métricas por UE ---
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
            #Encabezados
            headers = ["Receptor", "SINR eff(dB)", "SE(b/Hz)", "Shannon(b/Hz)", "SE vs Shannon(%)", "TBLER step",
                       "TBLER running"]
            line = "  ".join(f"{h:>14s}" for h in headers)
            sep = "-" * len(line)
            rows = [line, sep]

            #Mapeo y formateo
            def fmt(x, nd):
                try:
                    xf = float(x)
                    return f"{xf:.{nd}f}" if np.isfinite(xf) else "  NaN"
                except Exception:
                    return "  NaN"

            #Ordenar por id
            for m in sorted(ue_metrics, key=lambda x: x["ue_id"]):
                i = int(m["ue_id"])
                sinr = m.get("sinr_eff_db", float('nan'))
                se_la = m.get("se_la", float('nan'))
                se_sh = m.get("se_shannon", float('nan'))
                gap = m.get("se_gap_pct", float('nan'))
                tbler_step = m.get("tbler", float('nan'))

                #TBLER running
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

            legend_lines = []
            legend_lines.append("TBLER step: 0 = ACK, 1 = NACK, NaN = no agendado")
            legend_lines.append("TBLER running: 1 - ACK acum / TX acum")

            full_text = "\n".join(rows + ["", *legend_lines])
            self._ax_table_top.text(0.01, 0.98, full_text,
                                    va="top", ha="left", family="monospace", fontsize=9)

    def _render_to_figure(self):
        """Dibuja y actualiza la ventana gráfica (Modo Human)."""
        import matplotlib.pyplot as plt
        self._ensure_figure()
        self._render_common()
        self._fig.canvas.draw()
        self._fig.canvas.flush_events()
        plt.pause(1.0 / max(1, self.metadata.get("render_fps", 5)))

    def _render_to_array(self) -> np.ndarray:
        """
        Exporta el lienzo actual como un tensor RGB (H, W, 3).

        Crítico para el modo 'rgb_array' de Gymnasium, permitiendo que wrappers
        como `RecordVideo` capturen el entrenamiento del agente.
        """
        import numpy as np

        #Asegura figura y ejes, además pinta el contenido
        self._ensure_figure()
        self._render_common()

        #Dibuja en el canvas Agg
        self._fig.canvas.draw()

        #Tamaño en píxeles
        w, h = self._fig.canvas.get_width_height()

        #Buffer RGBA (bytes) -> ndarray (h, w, 4)
        buf = self._canvas.buffer_rgba()
        rgba = np.frombuffer(buf, dtype=np.uint8).reshape((h, w, 4))

        rgb = rgba[:, :, :3].copy()
        return rgb

    def _auto_view_2d(self, margin_ratio: float = 0.05):
        """
        Ajusta la escala de los ejes (vista 2D) basándose en el Bounding Box de la escena.
        Desactiva el auto-scaling dinámico para evitar que el mapa "tiemble" si el dron
        se mueve hacia los bordes.
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
        """Delegador del metodo render estándar de la API de Gymnasium."""
        if self.render_mode == "human":
            self._render_to_figure()
        elif self.render_mode == "rgb_array":
            return self._render_to_array()

    def close(self):
        """Limpieza profunda de memoria gráfica al cerrar el entorno."""
        import matplotlib.pyplot as plt
        if self._fig is not None:
            plt.close(self._fig)
        self._fig = self._ax = self._canvas = self._cbar = None
        self._ax_gp = None
        self._bars_gp = None
        self._bar_labels = []

    def _init_render_vars(self):
        """Inicializa punteros nulos para los componentes del UI."""
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
        self._acc = None
        self._last_ue_metrics = None

    def _get_metrics_info(self):
        """Extrae el diccionario de info auxiliar para la tupla (obs, reward, terminated, truncated, info)."""
        if self.run_metrics:
            #Modo Lento (Física + Métricas)
            sys_metrics = self.rt.run_sys_step()
            return {
                "ue_metrics": sys_metrics["ue_metrics"],
                "tbler_running_per_ue": sys_metrics.get("tbler_running_per_ue"),
            }
        #Modo Rápido (Física)
        return {"ue_metrics": [], "tbler_running_per_ue": []}

    def _handle_render(self, info):
        """Actualiza el estado interno antes de disparar el renderizado."""
        self._last_ue_metrics = info["ue_metrics"]
        self._last_tbler_running_per_ue = info.get("tbler_running_per_ue", None)
        if self.render_mode == "human":
            self._render_to_figure()
        elif self.render_mode == "rgb_array":
            info["frame"] = self._render_to_array()


    # ================= Render Profesional =================
    def render_dual_snapshot(self,
                                  prx_theory_dbm,
                                  prx_rt_dbm,
                                  title="Comparación de PRx: modelo teórico vs trazado de rayos",
                                  left_label="Modelo teórico (referencia)",
                                  right_label="Modelo por trazado de rayos (Sionna RT)",
                                  draw_links_theory=True,   #Líneas solo en modeloteórico
                                  draw_links_rt=False,      #Modelo RT sin líneas
                                  save=True,
                                  scene_pad_ratio=0.02):
        """
        Genera una visualización analítica comparando el modelo de Large-Scale Fading
        teórico (Log-Distance Pathloss) vs. el simulador electromagnético (Ray Tracing).

        Esta gráfica es ideal para validación científica, demostrando cómo el
        trazado de rayos captura sombras de edificios y desvanecimientos (Shadowing) que
        la teoría empírica simple ignora.
        """
        import numpy as np
        import matplotlib.pyplot as plt
        from pathlib import Path
        from matplotlib.colors import Normalize
        from matplotlib.cm import ScalarMappable
        from matplotlib.lines import Line2D
        from datetime import datetime

        #Inputs
        prx_theory_dbm = np.asarray(prx_theory_dbm, dtype=float).reshape(-1)
        prx_rt_dbm     = np.asarray(prx_rt_dbm,     dtype=float).reshape(-1)

        rx = np.asarray(self.mobility_manager.get_positions_xyz(), dtype=float)  #(N,3)
        N = rx.shape[0]
        assert prx_theory_dbm.size == N and prx_rt_dbm.size == N, \
            "PRx debe tener largo N (nº de receptores)."

        #Posición del dron
        pose = self.dron_Realista.get_pose()
        if isinstance(pose, (list, tuple)) and len(pose) > 0:
            pos = np.asarray(pose[0], dtype=float).reshape(-1)
        else:
            pos = np.asarray(pose, dtype=float).reshape(-1)

        drone_xyz = np.array([pos[0], pos[1], pos[2] if pos.size >= 3 else 0.0], dtype=float)
        h_tx = float(drone_xyz[2]) if np.isfinite(drone_xyz[2]) else np.nan

        # Distancia 3D
        d3d = np.linalg.norm(rx[:, :3] - drone_xyz[:3], axis=1)

        # Parámetros RF
        def _get_float(obj, name, default=np.nan):
            try:
                return float(getattr(obj, name, default))
            except Exception:
                return default

        fc_ghz = _get_float(self.rt, "freq_hz", np.nan) / 1e9

        pt_dbm = _get_float(self, "tx_power_dbm", np.nan)
        if np.isnan(pt_dbm):
            try:
                pt_dbm = float(self.rt._total_tx_power_dbm())
            except Exception:
                pt_dbm = np.nan

        rf_parts = []
        rf_parts.append(f"f={fc_ghz:.3f} GHz" if np.isfinite(fc_ghz) else "f=N/A")
        rf_parts.append(f"Pt={pt_dbm:.1f} dBm" if np.isfinite(pt_dbm) else "Pt=N/A")
        rf_str = " | ".join(rf_parts)

        #Bounds de escena (x/y) para grilla y márgenes
        def _get_scene_bounds_xy():
            sb = getattr(self, "scene_bounds", None)
            if sb is None and hasattr(self, "rt") and hasattr(self.rt, "scene_bounds"):
                sb = ((self.rt.scene_bounds[0][0], self.rt.scene_bounds[1][0]),
                    (self.rt.scene_bounds[0][1], self.rt.scene_bounds[1][1]),
                    (self.rt.scene_bounds[0][2], self.rt.scene_bounds[1][2]))
            if sb is None:
                return None
            (xmin, xmax), (ymin, ymax), _ = sb
            return float(xmin), float(xmax), float(ymin), float(ymax)

        def _apply_scene_bounds(ax, pad_ratio=0.02):
            xy = _get_scene_bounds_xy()
            if xy is None:
                return
            xmin, xmax, ymin, ymax = xy
            dx = (xmax - xmin) * pad_ratio
            dy = (ymax - ymin) * pad_ratio
            ax.set_xlim(xmin - dx, xmax + dx)
            ax.set_ylim(ymin - dy, ymax + dy)

        # Escala de color común
        vmin = float(np.nanmin([np.nanmin(prx_theory_dbm), np.nanmin(prx_rt_dbm)]))
        vmax = float(np.nanmax([np.nanmax(prx_theory_dbm), np.nanmax(prx_rt_dbm)]))
        norm = Normalize(vmin=vmin, vmax=vmax)
        cmap = "viridis"

        # Figura: 2 mapas y tabla
        fig = plt.figure(figsize=(14.5, 7.5), dpi=120)
        gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.42], hspace=0.10, wspace=0.18)

        axL = fig.add_subplot(gs[0, 0])
        axR = fig.add_subplot(gs[0, 1])
        axT = fig.add_subplot(gs[1, :])
        axT.axis("off")

        fig.suptitle(title, y=0.98, fontsize=14, weight="bold")
        fig.text(0.5, 0.945, rf_str, ha="center", va="center", fontsize=9)

        # Leyenda: altura del dron + receptores
        tx_label = f"Tx (Dron)  z={h_tx:.1f} m" if np.isfinite(h_tx) else "Tx (Dron)"
        legend_handles = [
            Line2D([0], [0], marker="^", linestyle="None", markerfacecolor="none",
                markeredgecolor="k", markersize=10, label=tx_label),
            Line2D([0], [0], marker="o", linestyle="None", markerfacecolor="none",
                markeredgecolor="k", markersize=8, label="Rx (Receptores)")
        ]

        # Panel plot
        def _plot_panel(ax, prx_dbm, panel_title, draw_links):
            # Líneas Tx->Rx (solo si se pide)
            if draw_links:
                for i in range(N):
                    ax.plot([drone_xyz[0], rx[i, 0]],
                            [drone_xyz[1], rx[i, 1]],
                            linestyle="-", linewidth=1.0, alpha=0.18, color="black")

            #Receptores (coloreados por PRx)
            ax.scatter(rx[:, 0], rx[:, 1],
                    c=prx_dbm, s=90, cmap=cmap, norm=norm,
                    edgecolors="k", linewidths=0.5)

            #Dron (TX)
            ax.scatter([drone_xyz[0]], [drone_xyz[1]],
                    marker="^", s=180, edgecolors="k",
                    facecolors="none", linewidths=1.2)

            #Etiquetas: Rx arriba y potencia abajo del punto
            for i in range(N):
                x, y = rx[i, 0], rx[i, 1]
                ax.text(x + 1.2, y + 1.2, f"Rx{i}", fontsize=8, weight="bold")
                ax.text(x + 1.2, y - 3.0, f"{prx_dbm[i]:.1f} dBm", fontsize=8)

            ax.set_title(panel_title, fontsize=12, weight="bold")
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlabel("x [m]")
            ax.set_ylabel("y [m]")

            #Aplicar límites de escena
            _apply_scene_bounds(ax, pad_ratio=scene_pad_ratio)

            ax.grid(True, alpha=0.25)
            ax.legend(handles=legend_handles, loc="upper right")

        _plot_panel(axL, prx_theory_dbm, left_label, draw_links_theory)
        _plot_panel(axR, prx_rt_dbm,     right_label, draw_links_rt)   # RT sin líneas

        #Colorbar única compartida
        sm = ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=[axL, axR], fraction=0.046, pad=0.02)
        cbar.set_label("PRx [dBm]")

        # Tabla inferior (SIN d2D)
        delta = prx_rt_dbm - prx_theory_dbm
        col_labels = ["Rx", "d3D [m]", "PRx teo [dBm]", "PRx RT [dBm]", "Δ [dB]"]

        cell_text = []
        for i in range(N):
            cell_text.append([
                f"Rx{i:02d}",
                f"{d3d[i]:.2f}",
                f"{prx_theory_dbm[i]:.2f}",
                f"{prx_rt_dbm[i]:.2f}",
                f"{delta[i]:+.2f}",
            ])

        table = axT.table(cellText=cell_text,
                        colLabels=col_labels,
                        loc="center",
                        cellLoc="center",
                        colLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.25)

        #Guardar figura
        if save:
            out_dir = Path("Environment-drones/figuras-comparacion-prx")
            out_dir.mkdir(parents=True, exist_ok=True)

            freq_suffix = f"{fc_ghz:.3f}GHz" if np.isfinite(fc_ghz) else "NA"
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = f"{timestamp}_prx_comp_teo_vs_rt_{freq_suffix}.png"
            fig.savefig(out_dir / filename, dpi=200, bbox_inches="tight")

        plt.show()
        return fig