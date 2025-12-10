from __future__ import annotations
import numpy as np
import time

# Importamos la clase principal de PyFlyt que controla la simulación
from PyFlyt.core import Aviary

class EntornoDronRealista:
    """
    Una clase que encapsula la simulación de un dron con física realista usando PyFlyt.
    Permite controlar el dron enviando comandos de velocidad.
    """
    def __init__(
        self,
        start_xyz: tuple[float, float, float] = (0.0, 0.0, 5.0),
        start_rpy: tuple[float, float, float] = (0.0, 0.0, 0.0),
        visualizar: bool = False,
    ):
        self.visualizar = visualizar
        
        self.aviary = Aviary(
            start_pos=np.array([start_xyz]), 
            start_orn=np.array([start_rpy]),
            render=self.visualizar,
            drone_type="quadx"
        )

        # Modo 6 es para control por VELOCIDAD
        self.aviary.set_mode(6) 
        
        print(f"Entorno de dron realista inicializado en la posición: {start_xyz}")

    def step(self, target_velocity_xyz: np.ndarray | list[float]):
        """
        Avanza un paso en la simulación aplicando una velocidad objetivo.
        """
        vx, vy, vz = target_velocity_xyz
        # Mapeamos [vx, vy, vz] al formato de acción [vx, vy, yaw_rate, vz]
        accion_para_pyflyt = np.array([vx, vy, 0.0, vz])
        
        # 1. Establecemos la acción para el dron
        self.aviary.set_actions(np.array([accion_para_pyflyt]))
        
        # 2. Avanzamos un paso en el motor de física
        self.aviary.step()

        if self.visualizar:
            time.sleep(1./240.)

        return self.get_position()

    def get_position(self) -> np.ndarray:
        """
        Obtiene la posición actual del dron en el simulador.
        """
        # --> CORRECCIÓN FINAL BASADA EN LA DOCUMENTACIÓN:
        # El atributo .state contiene toda la información.
        # Los 3 primeros elementos son las coordenadas (x, y, z).
        return self.aviary.drones[0].state[:3]

    def cerrar(self):
        """
        Cierra el entorno de simulación de forma segura.
        """
        self.aviary.close()
        print("Entorno cerrado.")

# --- EJEMPLO DE USO ---
if __name__ == "__main__":
    entorno = EntornoDronRealista(start_xyz=(0, 0, 5), visualizar=True)

    pos_actual = entorno.get_position()
    print(f"Posición inicial: {pos_actual}")

    print("\nIniciando simulación: Mover el dron hacia adelante y arriba...")
    
    velocidad_objetivo = np.array([5.0, 0.0, 2.0]) # vx=5, vy=0, vz=2
    
    for i in range(400):
        pos_actual = entorno.step(velocidad_objetivo)
        
        if i % 40 == 0:
            print(f"Paso {i}: Posición actual -> {pos_actual}")
            
        if i == 200:
            print("\nCambiando de dirección: Mover a la derecha y hacia abajo...")
            velocidad_objetivo = np.array([0.0, 5.0, -2.0]) # vx=0, vy=5, vz=-2

    entorno.cerrar()
    print(f"Simulación finalizada. Posición final: {pos_actual}")
    