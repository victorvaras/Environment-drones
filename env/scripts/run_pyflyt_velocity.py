"""
Demo de validación del wrapper DroneVelocityEnv con PyFlyt (modos 4, 6 y 7).

- Edita 'SELECT_MODE' para probar un modo concreto (4, 6 o 7).
- Con render=True verás PyBullet.
- Al final se muestran gráficas: 3D, XY, velocidades y altura.

Ejemplo de integración en tu Gym:
    pos, rpy = env.step_mode6([vx, vy, vr, vz], dt)
    # o
    pos, rpy = env.step_mode4([u, v, vr, z], dt)
    # o
    pos, rpy = env.step_mode7([x, y, r, z], dt)
"""
import time
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from env.environment.droneVelocityEnv import DroneVelocityEnv, DroneVelocityEnvConfig  # ajusta si cambias ruta
# Si corres este script directamente desde /mnt/data, puedes hacer:
# from drone_velocity_env import DroneVelocityEnv, DroneVelocityEnvConfig

OUT_DIR = Path.cwd() / "Environment drones" / "salidas_pyflyt"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    SELECT_MODE = 4  # cambia a 4 o 7 para probar otros

    cfg = DroneVelocityEnvConfig(
        start_xyz=(0.0, 0.0, 15.0),
        start_rpy=(0.0, 0.0, 0.0),
        control_hz=120,
        physics_hz=240,
        mode=SELECT_MODE,
        render=False,
        drone_model="cf2x",
        seed=42,
        record_trajectory=True,
    )
    env = DroneVelocityEnv(cfg)
    try:
        print("Pose inicial:", env.get_pose())

        if SELECT_MODE == 4:
            # Modo 4: [u, v, vr, z]  (u,v en cuerpo; z absoluto en mundo)
            seq4 = [
                ([0.0, 0.0, 0.0, 15.0], 5.0),     # hover en z=5
                #([5.0, 0.0, 0.0, 15.0], 5.0),     # avanzar en u
                #([0.0, 5.0, 0.0, 15.0], 5.0),     # derecha y subir a z=6.5
                #([0.0, 0.0, 0.0, 15.0], 5.0),     # girar en sitio
                #([-5.0, -5.0, 0.0, 15.0], 5.0),     # bajar a z=5
                #([0.0, 0.0, 0.0, 15.0], 5.0),
            ]
            respuesta = env.step_sequence_mode4(seq4)
            print("Respuesta secuencia modo 4:", respuesta[0][0])

        elif SELECT_MODE == 6:
            # Modo 6: [vx, vy, vr, vz]  (mundo). Con helper hold-z.
            z0 = 15.0
            env.step_mode6_holdz(vx=0.0, vy=0.0, vr=0.0, dt=5.0, z_ref=z0)   # estabiliza z
            env.step_mode6_holdz(vx=5.0, vy=0.0, vr=0.0, dt=5.0, z_ref=z0)   # recta en X
            env.step_mode6_holdz(vx=0.0, vy=5.0, vr=0.0, dt=5.0, z_ref=z0)  # recta Y + subir rápido
            env.step_mode6_holdz(vx=0.0, vy=0.0, vr=0.0, dt=5.0, z_ref=z0)  # giro en sitio
            env.step_mode6_holdz(vx=-5.0, vy=-5.0, vr=0.0, dt=5.0, z_ref=z0) # diagonal -X-Y + bajar rápido
            env.step_mode6_holdz(vx=0.0, vy=0.0, vr=0.0, dt=5.0, z_ref=z0)

        elif SELECT_MODE == 7:
            # Modo 7: [x, y, r, z] (setpoint de posición absoluta en mundo)
            seq7 = [
                ([0.0, 0.0, 0.0, 15.0], 10.0),     # ir a (0,0,15)
                ([25.0, 0.0, 0.0, 15.0], 10.0),     # ir a (5,0,15)
                ([25.0, 25.0, 0.0, 15.0], 10.0),     # ir a (5,5,15) con yaw≈1.2 rad
                ([25.0, 25.0, 0.0, 15.0], 10.0),
                ([0.0, 0.0, 0.0, 15.0], 10.0),     # volver hacia (2,2,15)
                ([0.0, 0.0, 0.0, 15.0], 10.0),
            ]
            env.step_sequence_mode7(seq7)

        else:
            raise ValueError("SELECT_MODE debe ser 4, 6 o 7.")

        print("Pose final:", env.get_pose())

        # --- Graficado ---
        #env.plot_trajectory_3d(show=True, save_path=None)
        #env.plot_xy(show=True, save_path=None, annotate_heading=True, stride=12)
        #env.plot_velocities(compare_cmd=True, show=True, save_path=None, title=f"Velocidades (modo {SELECT_MODE})")
        #env.plot_altitude(show=True, save_path=None, overlay_vz=True, smooth_window=5, title="Altura y v_z")

        # Guardar sin mostrar
        
        fecha = time.strftime("%Y%m%d_%H%M%S")
        NAME = f"{fecha}_modo_{SELECT_MODE}_demo"     # <- prefijo de archivos (cámbialo si quieres)

        files = env.save_all_plots(
            out_dir=OUT_DIR,
            name=NAME,
            annotate_heading=True,
            stride=12,
            include_vz=True,
            smooth_window=5,
        )

        print("Imágenes guardadas:")
        for k, v in files.items():
            print(f"  {k}: {v}")

    finally:
        print("Posición final (al cerrar):", env.get_pose())
        env.close()


if __name__ == "__main__":
    main()