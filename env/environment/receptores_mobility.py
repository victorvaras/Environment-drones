#Importaciones
import numpy as np
import torch
import socialforce
from socialforce import potentials

#Proyecto
from .spawn_manager import SpawnManager
from .receptores import ReceptoresManager, Receptor

class ReceptorMobilityManager:
    """
    Gestor centralizado de la movilidad de los receptores.
    Social Force Model, Control Reactivo y Validación Física.
    """

    def __init__(self, sionna_rt, bounds_min, bounds_max,
                 sfm_v0=5.0, sfm_sigma=0.5, sfm_u0=80.0, sfm_r=0.5):
        """
        Args:
            sionna_rt: Instancia de SionnaRT para validación de rayos y actualización de agentes.
            bounds_min/max: Límites de la escena para el SpawnManager.
            sfm_*: Parámetros físicos del modelo de fuerzas sociales.
        """

        #Parámetros Sionna y escena
        self.rt = sionna_rt
        self.bounds_min = bounds_min
        self.bounds_max = bounds_max

        #Parámetros SFM y SpawnManager
        #Parámetros Potencial Agente-Agente (PedPed)
        self.sfm_v0 = sfm_v0        #Fuerza de repulsión entre receptores
        self.sfm_sigma = sfm_sigma  #Alcance de la fuerza (radio personal)

        #Parámetros Potencial Agente-Obstáculo (PedSpace)
        self.sfm_u0 = sfm_u0  #Magnitud de la fuerza repulsiva del muro u obstáculo
        self.sfm_r = sfm_r    #Radio de influencia (distancia a la que el muro empieza a "empujar")

        #Estado interno
        self.sfm_sim = None             #Simulador SFM
        self.sfm_current_state = None   #Tensor de estado (PyTorch)
        self.sfm_goals_2d = None        #Metas 2D
        self.receptores_manager = None  #Manejador de receptores
        self.spawn_manager = None       #Spawn Manager
        self.obstacles_torch = None     #Tensor de obstáculos (PyTorch)

    def configure_obstacles(self, obstacles_np_list):
        """
        Recibe la lista de obstáculos del Slicer de Sionna.
        Configura el SpawnManager y prepara los tensores para SFM.
        """
        #1.Se configura SpawnManager
        self.spawn_manager = SpawnManager(obstacles_np_list, self.bounds_min, self.bounds_max)

        #2.Se preparan los obstáculos para SFM (Torch)
        #Se convierte la lista de Numpy a lista de Tensores PyTorch
        #Requisito de socialforce para cálculo vectorizado
        self.obstacles_torch = [torch.tensor(o, dtype=torch.float32) for o in obstacles_np_list]
        print(f"[ReceptorMobility] Obstáculos configurados: {len(self.obstacles_torch)} estructuras cargadas.")

    def reset(self, num_agents, rx_positions=None, rx_goals=None, seed=None):
        """
        Reinicia la simulación peatonal para un nuevo episodio.
        Genera posiciones/metas (si no se dan de manera estática) y reconstruye el simulador SFM.

        Returns:
            ReceptoresManager: La instancia con los agentes creados.
        """

        #Se recarga la semilla para la simulación
        if seed is not None:
            np.random.seed(seed)

        #Validar que el SpawnManager esté listo si vamos a necesitar generar algo
        if (rx_positions is None or rx_goals is None) and self.spawn_manager is None:
            raise RuntimeError("ERROR: Debes llamar a configure_obstacles() antes del reset dinámico.")

        #Márgenes de seguridad (Solo se usan si hay que generar las posiciones iniciales)
        #Margen con los obstáculos: para que no se creen ya sintiendo la fuerza del muro
        safe_wall_dist = self.sfm_r * 1.5

        #Margen entre receptores: para que no se creen ya sintiendo la fuerza de otro receptor
        safe_agent_dist = self.sfm_sigma * 2.2

        #Lógica de posiciones iniciales de receptores
        if rx_positions is None:
            #Caso 1: Generación Dinámica (No se entregan posiciones)
            rx_positions = self.spawn_manager.generate_positions(
                n_agents=num_agents,             #Número de receptores
                min_dist_obs=safe_wall_dist,     #Distancia segura contra obstáculos
                min_dist_agents=safe_agent_dist, #Distancia segura entre receptores
                z_height=1.5                     #Altura para receptores
            )
            #Se actualiza el número real de agentes generados (por si es que el SpawnManager genera menos)
            current_n = len(rx_positions)
        else:
            #Caso 2: Posiciones Manuales (Estáticas)
            #Se confia en las posiciones que son entregadas
            current_n = len(rx_positions)

        #Lógica de metas de receptores
        #Se realiza de manera independiente.
        #Esto dado que si faltan metas (aunque haya posiciones manuales), se generan.
        #Generar Metas (Goals) si no existen
        if rx_goals is None:
            rx_goals = self.spawn_manager.generate_positions(
                n_agents=current_n,             #Número de receptores
                min_dist_obs=safe_wall_dist,    #Distancia segura contra obstáculos
                min_dist_agents=0.5,            #Distancia segura entre receptores (para evitar metas muy juntas)
                z_height=1.5                    #Altura para receptores
            )

        #Ahora que se tienen las posiciones iniciales (manuales o generadas), se crean los objetos
        self.receptores_manager = ReceptoresManager([Receptor(*p) for p in rx_positions])

        #Configuración Motor SFM
        dt = 0.1  #Paso de tiempo físico (100 ms)

        #Potencial Agente-Agente (PedPed)
        #Define la fuerza de repulsión entre peatones para evitar choques entre ellos.
        ped_ped = potentials.PedPedPotential2D(
            v0=self.sfm_v0,       #Fuerza de repulsión entre receptores
            sigma=self.sfm_sigma, #Alcance de la fuerza
            asymmetry=0.3         #Factor de asimetria para esquive vertical entre receptores
        )
        ped_ped.delta_t_step = dt

        #Potencial Agente-Obstáculo (PedSpace)
        ped_space = potentials.PedSpacePotential(
            self.obstacles_torch, #Se inyecta el mapa de obstáculos extraído de Sionna.
            u0=self.sfm_u0,       #Magnitud de la fuerza repulsiva del muro u obstáculo
            r=self.sfm_r          #Radio de influencia (distancia a la que el muro empieza a "empujar")
        )

        #Inicialización del Simulador SFM
        #Este gestionará el movimiento autónomo de los peatones.
        self.sfm_sim = socialforce.Simulator(
            delta_t=dt,             #Paso de tiempo dentro del simulador SFM
            ped_ped=ped_ped,        #Potencial Agente-Agente (PedPed)
            ped_space=ped_space,    #Potencial Agente-Obstáculo (PedSpace)
            oversampling=1,         #Sincroniza la física del SFM 1:1 con el paso de Gym.
            field_of_view=-1        #Permite que el Campo de visión sea de 360° (APF para cada receptor).
        )
        self.sfm_sim.ped_space = ped_space
        #Referencia Externa para el Script
        #Se guarda una referencia al potencial del PedSpace para visualización/debug

        #Construcción del Tensor de Estado Inicial (State Tensor o Tensor de estado)
        #Formato requerido por SFM: [x, y, vx, vy, ax, ay, gx, gy, tau, v_pref]
        np_initial = np.array(rx_positions) #Posiciones iniciales
        np_goals = np.array(rx_goals)       #Metas (Goals)

        #Se guardan las metas 2D para la lógica de control reactivo
        self.sfm_goals_2d = np_goals[:, :2]

        #Inicialización de variables
        n = current_n #
        initial_state = np.hstack([
            np_initial[:, :2],                    #Columna 0-1: Posición inicial (x, y)
            np.zeros((n, 2)),                     #Columna 2-3: Velocidad inicial (vx, vy)
            np.zeros((n, 2)),                     #Columna 4-5: Aceleración inicial (ax, ay)
            self.sfm_goals_2d,                    #Columna 6-7: Metas (Goals) (gx, gy)
            np.full((n, 1), 0.5),  #Columna 8  : Tiempo de reacción (Tau)
            np.full((n, 1), 1.3)   #Columna 9  : Velocidad Preferida (v_pref = 1.3 m/s)
        ])

        #Se convierte a Tensor de PyTorch (formato requerido para la librería socialforce)
        self.sfm_current_state = torch.tensor(initial_state, dtype=torch.float32)

        return self.receptores_manager

    def step(self, dt=0.1):
        """
        Ejecuta UN paso de simulación completo:
        1.SFM (Cálculo de fuerzas sociales)
        2.Control Reactivo (Anti-Atascos) (Corrección de mínimos locales y estancamiento.)
        3.Validación Sionna (Colisiones duras)
        4.Cálculo de Velocidad Real (Doppler)
        """
        if self.sfm_sim is None:
            return

        #1.Obtener propuesta del SFM
        #Se consulta a la librería socialforce el siguiente estado deseado.
        #Se considera: Atracción a la meta + Repulsión entre agentes + Repulsión de obstáculos.
        proposed_state = self.sfm_sim(self.sfm_current_state) #Se obtiene el "estado propuesto" por el SFM
        proposed_state_np = proposed_state.detach().numpy()   #Se convierte la respuesta de Torch a NumPy

        #Se extrae solo el vector velocidad propuesto (Vx, Vy)
        proposed_vel_2d = proposed_state_np[:, 2:4]

        #Se obtiene la posición actual
        current_pos_2d = self.sfm_current_state.numpy()[:, 0:2]

        #2.Control Reactivo: Evasión de Mínimos Locales
        #Problema: El modelo SFM puede caer en equilibrios estables (velocidad ~0) frente a muros planos.
        #Solución: Se detecta el estancamiento y se aplica una fuerza de escape tangencial.
        for i in range(self.receptores_manager.n):
            #Análisis del Estado Cinético (Velocidad actual)
            vel_mag = np.linalg.norm(proposed_vel_2d[i])

            #Análisis Geométrico (Vector a la Meta)
            goal_vec = self.sfm_goals_2d[i] - current_pos_2d[i]
            dist_goal = np.linalg.norm(goal_vec)

            #Algoritmo de Detección de Estancamiento (Stuck Detection)
            #Se asume que está bloqueado por un equilibrio de fuerzas.
            if dist_goal > 1.0 and vel_mag < 0.05:
                #Evasión Tangencial (Maniobra de Recuperación)
                #Se normaliza el vector dirección
                to_goal_dir = goal_vec / dist_goal if dist_goal > 1e-3 else np.zeros(2)

                #Se calcula un vector perpendicular a la meta (-y, x) para generar deslizamiento lateral.
                #De tal manera que se rompe la simetría y fuerza al agente a rodear el obstáculo (Wall Sliding).
                #Vector base perpendicular (90 grados respecto a la meta)
                tangent = np.array([-to_goal_dir[1], to_goal_dir[0]])

                #Determinismo por ID del receptor (Par/Impar)
                #Se usa el índice 'i' del agente.
                #Esto garantiza que el agente nunca cambie de opinión a mitad de camino.
                #Si i es par -> Gira a un lado. Si es impar -> Gira al otro.
                if i % 2 == 0:
                    tangent = -tangent

                #Inyección de Fuerza
                proposed_vel_2d[i] += tangent * 8.0

        #3.Integración de Euler (Cálculo de nueva posición)
        #Se calcula la posición candidata: Posición Nueva = Posición Actual + Velocidad Final * dt
        final_proposed_pos_2d = current_pos_2d + proposed_vel_2d * dt

        #4.Validación física, actualización y cálculo de Velocidad Real (Doppler Logic)
        #Se prepara el nuevo array de estado validado y actualización
        new_validated_state_np = self.sfm_current_state.numpy().copy()

        #Se obtienen posiciones reales actuales desde Sionna
        current_pos_3d = np.array([rx.position for rx in self.rt.rx_list], dtype=np.float64)

        for i, rx in enumerate(self.rt.rx_list):
            #Posición Anterior (t)
            prev_x = float(current_pos_3d[i][0])  #Coordenada X
            prev_y = float(current_pos_3d[i][1])  #Coordenada Y
            z_fixed = float(current_pos_3d[i][2]) #Coordenada Z

            #Posición Candidata (t + dt)
            next_x = float(final_proposed_pos_2d[i, 0]) #Coordenada X
            next_y = float(final_proposed_pos_2d[i, 1]) #Coordenada Y

            #Se construyen las coordenadas candidatas
            p_curr = np.array([prev_x, prev_y, z_fixed]) #Posición actual
            p_next = np.array([next_x, next_y, z_fixed]) #Posición propuesta

            #Chequeo de Colisión Dura (Hard Collision)
            #Si a pesar de las fuerzas de repulsión, el agente intenta atravesar un muro u obstáculo
            #el motor de física (Sionna) lo detiene.
            if self.rt.is_move_valid_receptores(p_curr, p_next):
                #Movimiento Válido: se actualiza la física (posición) y estado
                rx.position = [next_x, next_y, z_fixed]

                #Se cálcula la velocidad real (v = (pos_final - pos_inicial) / dt)
                #Velocidad Real = Distancia / Tiempo (Perfecta para Doppler)
                #Esto incluye magnitud exacta y signo correcto automáticamente.
                real_vx = (next_x - prev_x) / dt
                real_vy = (next_y - prev_y) / dt
                rx.velocity = [real_vx, real_vy, 0.0]

                #Se actualiza el estado interno para el siguiente step del SFM
                new_validated_state_np[i, 0:2] = [next_x, next_y]
                new_validated_state_np[i, 2:4] = [real_vx, real_vy]
            else:
                #Colisión: Detención de emergencia
                rx.velocity = [0.0, 0.0, 0.0]
                new_validated_state_np[i, 0:2] = [prev_x, prev_y]
                new_validated_state_np[i, 2:4] = [0.0, 0.0]

        #5.Se guarda y actualizar el Tensor SFM del estado para ser utilizado el siguiente ciclo
        self.sfm_current_state = torch.tensor(new_validated_state_np, dtype=torch.float32)