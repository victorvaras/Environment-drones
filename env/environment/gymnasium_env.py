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
from .spawn_manager import SpawnManager

class DroneEnv(gym.Env):
    """Entorno Gymnasium con Sionna RT (sin imagen de fondo)."""
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 15}

    def __init__(
            self,
            rx_positions: list[tuple[float, float, float]] | None = None,
            rx_goals: list[tuple[float, float, float]] | None = None,
            num_agents: int = 10,
            frequency_mhz: float = 3500.0,
            tx_power_dbm: float = 30.0,
            bandwidth_hz: float = 20e6,
            scene_name: str = "munich",
            antenna_mode: str = "ISO",
            max_steps: int = 400,
            render_mode: str | None = None,
            drone_start: tuple[float, float, float] = (0.0, 0.0, 20.0),
            run_metrics: bool = False

    ):
        super().__init__()
        assert render_mode in (None, "human", "rgb_array"), \
            "render_mode debe ser None, 'human' o 'rgb_array'"
        self.render_mode = render_mode

        self._start = drone_start
        self.max_steps = int(max_steps)
        self.step_count = 0
        self.run_metrics = run_metrics

        # --- 1. Configuración inicial de lo receptores ---
        #Si se le asignan posiciones de manera estática
        if rx_positions is not None:
            self.current_num_agents = len(rx_positions)
            using_manual_spawn = True #Se utilizara de manera manual
        else:
            self.current_num_agents = num_agents
            using_manual_spawn = False #Se utilizara el SpawnManager

        #Se definen las constantes físicas para usarlas en SpawnManager y SFM
        #Parámetros Potencial Agente-Agente (PedPed)
        self.sfm_v0 = 5.0     #Fuerza de repulsión entre receptores
        self.sfm_sigma = 0.5  #Alcance de la fuerza (radio personal)

        #Parámetros Potencial Agente-Obstáculo (PedSpace)
        self.sfm_u0 = 80.0 #Magnitud de la fuerza repulsiva del muro u obstáculo
        self.sfm_r = 0.5   #Radio de influencia (distancia a la que el muro empieza a "empujar")

        #Se inicializan las velocidades en 0 para el cálculo de Doppler
        rx_velocities_mps = [(0.0, 0.0, 0.0) for _ in range(self.current_num_agents)]

        # --- 2. SIONNA RT ---
        #Se cargar la escena antes de configurar la física para saber dónde están los obstáculos.
        self.rt = SionnaRT(
            antenna_mode=antenna_mode,
            frequency_mhz=frequency_mhz,
            # tx_power_dbm=tx_power_dbm,
            # bandwidth_hz=bandwidth_hz,
            scene_name=scene_name,
            num_ut=self.current_num_agents, #Se reserva memoria para N receptores
            rx_velocities_mps=rx_velocities_mps,
        )

        #Se construye la escena en Sionna
        self.rt.build_scene()

        # --- 3. PUENTE DE DATOS: EXTRACCIÓN DE OBSTÁCULOS (SLICER) ---
        #Se extrae la geometría estática de Sionna una sola vez.
        #Se utiliza la función 'get_sfm_obstacles'.
        print(f"[Gym] Extrayendo obstáculos de la escena '{scene_name}'...")

        # --- LÓGICA DE AUTO-ESCALADO (AUTO-SCALE) ---
        #Se calcula el tamaño del mapa para decidir la densidad y proteger la RAM.
        bounds = self.rt.mi_scene.bbox()
        extent = max(bounds.max.x - bounds.min.x, bounds.max.y - bounds.min.y)

        if extent > 1000.0:
            gym_density = 1.5  #Escena Grande -> Menos resolución
            print(f"[Gym] Escena Gigante ({extent:.0f}m). Ajustando densidad a: {gym_density}m")
        elif extent > 500.0:
            gym_density = 0.8  #Escena mediana
            print(f"[Gym] Escena Grande ({extent:.0f}m). Ajustando densidad a: {gym_density}m")
        else:
            gym_density = 0.4  #Escena pequeña -> Alta precisión
            print(f"[Gym] Escena Estándar ({extent:.0f}m). Usando alta precisión: {gym_density}m")

        #Se utiliza el escaner para obtener los obstáculos para la API Socialforce (Slicer)
        #grid_density = densidad calculada dinámicamente
        obstacles_np = self.rt.get_sfm_obstacles(grid_density=gym_density)

        #Se convierte la lista de Numpy a lista de Tensores PyTorch
        #Requisito de socialforce para cálculo vectorizado
        obstacles_torch = [torch.tensor(o, dtype=torch.float32) for o in obstacles_np]
        print(f"[Gym] Se encontraron {len(obstacles_torch)} estructuras de obstáculos.")

        # --- 4. GENERACIÓN DINÁMICA DE POSICIONES (SPAWN MANAGER) ---
        #Si no se entregan posiciones manuales, se generan usando los obstáculos encontrados
        if not using_manual_spawn:
            print(f"[Gym] Generando {self.current_num_agents} posiciones y metas dinámicas")

            #Límites 2D para el generador de posiciones
            b_min = (self.rt.scene_bounds[0][0], self.rt.scene_bounds[0][1])
            b_max = (self.rt.scene_bounds[1][0], self.rt.scene_bounds[1][1])

            #Se realiza el llamado al metodo SpawnManager
            spawn_manager = SpawnManager(obstacles_np, b_min, b_max)

            #--- Vinculación física con SFM ---
            #Margen entre receptores: sigma * 2.2 para que no se creen ya sintiendo la fuerza de otro receptor
            safe_agent_dist = self.sfm_sigma * 2.2
            #Margen de seguridad: r * 1.1 para que no se creen ya sintiendo la fuerza del muro
            safe_wall_dist = self.sfm_r * 1.5

            #Generar Posiciones Iniciales
            rx_positions = spawn_manager.generate_positions(
                n_agents=self.current_num_agents, #Número de receptores
                min_dist_obs=safe_wall_dist,      #Distancia segura contra obstáculos
                min_dist_agents=safe_agent_dist,  #Distancia segura entre receptores
                z_height=1.5                      #Altura para receptores
            )

            #Generar Metas (Goals) si no existen
            if rx_goals is None:
                rx_goals = spawn_manager.generate_positions(
                    n_agents=self.current_num_agents, #Número de receptores
                    min_dist_obs=safe_wall_dist,      #Distancia segura contra obstáculos
                    min_dist_agents=0.5,              #Distancia segura entre receptores (para evitar metas muy juntas)
                    z_height=1.5                      #Altura para receptores
                )

        # --- 5. Confirmación de receptores ---
        #Ahora que se tienen las posiciones iniciales (manuales o generadas), se crean los objetos
        self.receptores = ReceptoresManager([Receptor(*p) for p in rx_positions])
        numero_receptores = self.receptores.n

        #Se actualizan en Sionna las posiciones iniciales
        self.rt.attach_receivers(self.receptores.positions_xyz())

        # --- 6. CONFIGURACIÓN MOTOR DE NAVEGACIÓN Socialforce - SFM ---
        dt = 0.1 #Paso de tiempo físico (100ms)

        #Potencial Agente-Agente (PedPed)
        #Define la fuerza de repulsión entre peatones para evitar choques entre ellos.
        ped_ped_potential = potentials.PedPedPotential2D(
            v0=self.sfm_v0,       #Fuerza de repulsión entre receptores
            sigma=self.sfm_sigma, #Alcance de la fuerza
            asymmetry=0.3         #Factor de asimetria para esquive vertical entre receptores
        )
        ped_ped_potential.delta_t_step = dt

        #Potencial Agente-Obstáculo (PedSpace)
        ped_space_potential = potentials.PedSpacePotential(
            obstacles_torch,  #Se inyecta el mapa de obstáculos extraído de Sionna.
            u0=self.sfm_u0,  #Magnitud de la fuerza repulsiva del muro u obstáculo
            r=self.sfm_r     #Radio de influencia (distancia a la que el muro empieza a "empujar")
        )

        #Inicialización del Simulador SFM
        #Este gestionará el movimiento autónomo de los peatones.
        self.sfm_sim = socialforce.Simulator(
            delta_t=dt,                     #Paso de tiempo dentro del simulador SFM
            ped_ped=ped_ped_potential,      #Potencial Agente-Agente (PedPed)
            ped_space=ped_space_potential,  #Potencial Agente-Obstáculo (PedSpace)
            oversampling=1,                 #Sincroniza la física del SFM 1:1 con el paso de Gym.
            field_of_view=-1                #Permite que el Campo de visión sea de 360° (APF para cada receptor).
        )

        #Referencia Externa para el Script
        #Se guarda una referencia al potencial del PedSpace para visualización/debug
        self.sfm_sim.ped_space = ped_space_potential

        # --- 7. Construcción del Tensor de Estado Inicial (State Tensor o Tensor de estado) ---
        #Formato requerido por SFM: [x, y, vx, vy, ax, ay, gx, gy, tau, v_pref]
        np_initial_states = np.array(rx_positions)  #Posiciones iniciales
        np_goal_states = np.array(rx_goals)         #Metas (Goals)

        #Se guardan las metas 2D para la lógica de control
        self.sfm_goals_2d = np_goal_states[:, :2]

        #Inicialización de variables
        initial_velocities_np = np.zeros((numero_receptores, 2))                    #Velocidades iniciales (vx, vy)
        initial_accelerations_np = np.zeros((numero_receptores, 2))                 #Aceleraciones iniciales (ax, ay)
        initial_taus_np = np.full((numero_receptores, 1), 0.5)       #Tiempo de reacción (tau)
        preferred_speeds_np = np.full((numero_receptores, 1), 1.3)   #Velocidad deseada (v_pref = 1.3 m/s)

        #Se ensambla el Tensor de Estado
        initial_state_np = np.hstack([
            np_initial_states[:, :2],   #Columna 0-1: Posición (x, y)
            initial_velocities_np,      #Columna 2-3: Velocidad (vx, vy)
            initial_accelerations_np,   #Columna 4-5: Aceleración (ax, ay)
            self.sfm_goals_2d,          #Columna 6-7: Metas (Goals) (gx, gy)
            initial_taus_np,            #Columna 8  : Tau
            preferred_speeds_np         #Columna 9  : Velocidad Preferida (v_pref)
        ])

        #Se convierte a Tensor de PyTorch (formato requerido para la librería socialforce)
        self.sfm_current_state = torch.tensor(initial_state_np, dtype=torch.float32)

        # --- 8. Configuración Final del Entorno (Bounds, Spaces, Rendering) ---
        #Se definen los límites del espacio de acción y observación para RL
        bounds_min = self.rt.scene_bounds[0]
        bounds_max = self.rt.scene_bounds[1]
        bounds = ((bounds_min[0], bounds_max[0]), (bounds_min[1], bounds_max[1]), (bounds_min[2], bounds_max[2]))
        self.scene_bounds = bounds

        #Inicialización del Dron
        self.dron = Dron(start_xyz=self._start, bounds=bounds)
        self.rt.move_tx(self.dron.pos)  #Sincronización inicial física-lógica

        #Espacios de Gymnasium (Acción y Observación)
        self.action_space = spaces.Box(low=-5.0, high=5.0, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-1e9, high=1e9, shape=(3 + self.receptores.n,), dtype=np.float32
        )
        self.dron.bounds = bounds

        #Estado de renderizado
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


    def reset(self, *, seed: int | None = None, options: dict | None = None):
        #Reinicia el entorno a su estado inicial para un nuevo episodio.
        super().reset(seed=seed)
        self.step_count = 0

        #Manejar semilla en Spawn Manager
        if seed is not None:
            np.random.seed(seed)

        #Reinicio del Dron
        self.dron = Dron(start_xyz=self._start, bounds=self.scene_bounds)

        #Sincronización con Sionna
        #Se mueve el transmisor a la posición inicial para que el cálculo sea correcto desde t=0
        self.rt.move_tx(self.dron.pos)

        #Generación de la primera observación
        obs = np.concatenate([self.dron.pos]).astype(np.float32)
        info = {}

        #Limpieza de estado de renderizado
        #Se borran las referencias gráficas para evitar superposiciones en nuevos episodios
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

        #Verificación auxiliar del número de usuarios
        try:
            self.num_ut = int(self.receptores.positions_xyz().shape[0])
        except Exception:
            # Fallback si tu contenedor expone otra API
            self.num_ut = int(getattr(self.receptores, "num", getattr(self.receptores, "n", 0)))

       

        return obs, info

    def step(self, action: np.ndarray):
        #Ejecuta un paso de simulación (t -> t+dt)
        """
        Para el movimiento de los receptores o personas se implemento una
        arquitectura de navegación híbrida:
        1. Planificación Local (SFM): Cálculo de fuerzas sociales base.
        2. Control Reactivo (Python): Corrección de mínimos locales y estancamiento.
        """
        self.step_count += 1

        # Movimiento del dron
        self.dron.step_delta(action)
        self.rt.move_tx(self.dron.pos)

        #---Movimiento de personas / receptores (Rx's) - Social Force Model (SFM)---
        #Se consulta a la librería socialforce el siguiente estado deseado.
        #Se considera: Atracción a la meta + Repulsión entre agentes + Repulsión de obstáculos.
        proposed_state = self.sfm_sim(self.sfm_current_state) #Se obtiene el "estado propuesto" por el SFM
        proposed_state_np = proposed_state.detach().numpy()   #Se convierte la respuesta de Torch a NumPy

        #Se extrae solo el vector velocidad propuesto (Vx, Vy)
        proposed_vel_2d = proposed_state_np[:, 2:4]

        # ------------------------------------------------------------
        # Control Reactivo: Evasión de Mínimos Locales
        # ------------------------------------------------------------
        #Problema: El modelo SFM puede caer en equilibrios estables (velocidad ~0) frente a muros planos.
        #Solución: Se detecta el estancamiento y se aplica una fuerza de escape tangencial.
        current_pos_2d = self.sfm_current_state.numpy()[:, 0:2]

        for i in range(self.receptores.n):
            #Análisis del Estado Cinético (Velocidad actual)
            vel_magnitude = np.linalg.norm(proposed_vel_2d[i])

            #Análisis Geométrico (Vector a la Meta)
            goal_pos = self.sfm_goals_2d[i]
            to_goal_vec = goal_pos - current_pos_2d[i]
            dist_to_goal = np.linalg.norm(to_goal_vec)

            #Se normaliza el vector dirección
            if dist_to_goal > 1e-3:
                to_goal_dir = to_goal_vec / dist_to_goal
            else:
                to_goal_dir = np.array([0.0, 0.0])

            #Algoritmo de Detección de Estancamiento (Stuck Detection)
            #Criterio: Si el agente está lejos de su meta (>1.0m) pero su velocidad es casi nula (<0.2),
            #se asume que está bloqueado por un equilibrio de fuerzas.
            if dist_to_goal > 1.0 and vel_magnitude < 0.2:
                #Maniobra de Recuperación (Tangential Escape)
                #Se calcula un vector perpendicular a la meta (-y, x) para generar deslizamiento lateral.
                #De tal manera que se rompe la simetría y fuerza al agente a rodear el obstáculo (Wall Sliding).
                #Vector base perpendicular (90 grados respecto a la meta)
                tangent_force = np.array([-to_goal_dir[1], to_goal_dir[0]])

                #Determinismo por ID del receptor
                #Se usa el índice 'i' del agente.
                #Esto garantiza que el agente nunca cambie de opinión a mitad de camino.
                #Si i es par -> Gira a un lado. Si es impar -> Gira al otro.
                if i % 2 == 0:
                    tangent_force = -tangent_force

                #Inyección de Fuerza
                #Se usa 8.0 para que la fuerza de escape sea comparable a la del muro (en este caso u0 = 80).
                #asegurando un deslizamiento rápido y visible.
                proposed_vel_2d[i] += tangent_force * 8.0
                # ============================================================

        #Integración de Euler (Cálculo de nueva posición)
        #Se calcula la posición candidata: Posición Nueva = Posición Actual + Velocidad Final * dt
        dt = 0.1
        final_proposed_pos_2d = current_pos_2d + proposed_vel_2d * dt

        #Validación física y actualización
        #Se prepara el nuevo array de estado validado y actualización
        current_positions_3d = np.array([rx.position for rx in self.rt.rx_list])
        new_validated_state_np = self.sfm_current_state.numpy().copy()
        #Esto ayuda evita el atravesar muros debido a pasos de tiempo discretos.
        #Se consulta si el movimiento es válido

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
        #sys_metrics = self.rt.run_sys_step()
        if self.run_metrics:
            # Modo Lento (Completo)
            sys_metrics = self.rt.run_sys_step()
            info = {
                "ue_metrics": sys_metrics["ue_metrics"],
                "tbler_running_per_ue": sys_metrics.get("tbler_running_per_ue"),
            }
        else:
            # Modo Rápido (Solo Física)
            info = {
                "ue_metrics": [],
                "tbler_running_per_ue": [],
            }

        # --- Recompensa ---
        # reward = float(np.mean(snr))
        reward = 1.0

        # --- Observación ---
        obs = np.concatenate([self.dron.pos]).astype(np.float32)

        # --- Terminación ---
        terminated = False
        truncated = self.step_count >= self.max_steps

        # Renderizado (usa 'info', que ya está seguro)
        if self.render_mode == "human":
            self._last_ue_metrics = info["ue_metrics"]
            self._last_tbler_running_per_ue = info.get("tbler_running_per_ue", None)
            self._render_to_figure()
        elif self.render_mode == "rgb_array":
            self._last_ue_metrics = info["ue_metrics"]
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