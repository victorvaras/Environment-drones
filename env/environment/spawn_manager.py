#Importaciones
import numpy as np
from scipy.spatial import cKDTree

class SpawnManager:
    """
    Gestor de posiciones iniciales y metas para receptores.

    Con este metodo se garantiza:
    1.No colisión con obstáculos (usando KD-Tree sobre la nube de puntos del Slicer).
    2.No colisión entre agentes o receptores.
    3.Altura Z fija (para simulación).
    """

    def __init__(self, obstacles_np_list, bounds_min, bounds_max):
        """
        Args:
            obstacles_np_list: Lista de arrays (N,2) que vienen del Slicer (get_sfm_obstacles).
            bounds_min: Tupla (x_min, y_min) de la escena.
            bounds_max: Tupla (x_max, y_max) de la escena.
        """
        #Se aplana la lista de obstáculos en una sola matriz para su uso en KD-Tree
        if obstacles_np_list and len(obstacles_np_list) > 0:
            self.all_obstacles = np.vstack(obstacles_np_list)
            #Se crea el árbol de búsqueda espacial (KD-Tree) para realizar consultas rápidas
            self.tree = cKDTree(self.all_obstacles)
        else:
            self.all_obstacles = None
            self.tree = None

        #Se asignan los valores de la escena
        self.bounds_min = np.array(bounds_min)
        self.bounds_max = np.array(bounds_max)

    #Generador de posiciones
    def generate_positions(self,
                           n_agents: int, #Número de receptores
                           min_dist_obs: float,  #Distancia   con los obstáculos (basado en 'r' de SFM)
                           min_dist_agents: float,  #Distancia mínima entre receptores (basada en 'sigma' de SFM)
                           z_height: float = 1.5,  #Altura fija de los receptores
                           max_retries: int = 10000):

        #Puntos válidos para posiciones e intentos
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
                    continue  #Se rechaza, dado que esta muy cerca de otra receptor (viola radio 'sigma')

            #4.Aceptación: se le agrega la altura Z fija y es agregado a los puntos válidos
            valid_points.append((float(rand_xy[0]), float(rand_xy[1]), float(z_height)))

        return valid_points