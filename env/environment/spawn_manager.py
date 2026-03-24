#Importaciones
import numpy as np
from scipy.spatial import cKDTree

class SpawnManager:
    """
    Módulo de gestión de posiciones iniciales y metas para receptores (UE).

    Este componente revisa la topología del escenario antes de iniciar la simulación,
    asegurando que los receptores cumplan con las condiciones y restricciones fisicas del simulador.
    """

    def __init__(self, obstacles_np_list, bounds_min, bounds_max):
        """
        Se inicializa el gestor espacial de obstáculos.

        Args:
            obstacles_np_list: Nube de puntos proveniente del Slicer (Lista de arrays (N,2)).
                               Define la geometría de los obstaculos del escenario.
            bounds_min/max: Límites del espacio en metros. (x_min, y_min) y (x_max, y_max) de la escena.
        """
        #Se consolida la geometría de obstáculos (lista de obstáculos) en una matriz global (N, 2).
        if obstacles_np_list and len(obstacles_np_list) > 0:
            self.all_obstacles = np.vstack(obstacles_np_list)

            #Se implementa el árbol de búsqueda espacial (cKD-Tree) para realizar consultas y búsquedas rápidas.
            self.tree = cKDTree(self.all_obstacles)
        else:
            self.all_obstacles = None
            self.tree = None

        #Se asignan los valores de los límites de la escena
        self.bounds_min = np.array(bounds_min)
        self.bounds_max = np.array(bounds_max)

    def generate_positions(self,
                           n_agents: int,
                           min_dist_obs: float,
                           min_dist_agents: float,
                           z_height: float = 1.5,
                           max_retries: int = 10000):
        """
        Genera coordenadas cartesianas [x, y, z] para los receptores.
        La lógica de rechazo asegura que los receptores no inicien en estados de
        colisión inminente, lo que desestabilizaría las fuerzas del SFM.

        Args:
            n_agents: Número de receptores.
            min_dist_obs: Distancia con los obstáculos (basado en 'r' del SFM).
            min_dist_agents: Distancia mínima entre receptores (basada en 'sigma' del SFM).
            z_height: Altura fija de los receptores.
            max_retries: Cantidad máxima de reintentos en la búsqueda de posiciones.
        """

        #Coordenadas o puntos válidos para posiciones e intentos de búsqueda
        valid_points = []
        attempts = 0

        while len(valid_points) < n_agents:
            if attempts > max_retries:
                print(
                    f"[SpawnManager]: Solo se pudieron generar {len(valid_points)}/{n_agents} posiciones válidas después de {attempts} intentos.")
                break

            #1.Se genera candidato aleatorio [x, y] dentro de los límites de la escena
            rand_xy = np.random.uniform(self.bounds_min, self.bounds_max)
            attempts += 1  #Se incrementa el número del intento

            #2.Primera validación: ¿Está muy cerca de un obstáculo?
            if self.tree is not None:
                #Se devuelve (distancia, índice), solo se verifica la distancia.
                dist_obs, _ = self.tree.query(rand_xy, k=1)
                if dist_obs < min_dist_obs:
                    continue  #Se rechaza, dado que esta muy cerca de un obstáculo (viola radio 'r')

            #3.Segunda validación: ¿Está muy cerca de otro receptor ya creado?
            if len(valid_points) > 0:
                #Se extraen solo las coordenadas X,Y de los puntos ya validados
                existing_xy = np.array(valid_points)[:, :2]
                #Se calcula la distancia euclidiana a todos ellos
                dists_agents = np.linalg.norm(existing_xy - rand_xy, axis=1)

                if np.min(dists_agents) < min_dist_agents:
                    continue  #Se rechaza, dado que esta muy cerca de otro receptor (viola radio 'sigma')

            #4.Aceptación: si cumple con las dos validaciones se le agrega la altura Z fija y es agregado a los puntos válidos
            valid_points.append((float(rand_xy[0]), float(rand_xy[1]), float(z_height)))

        return valid_points