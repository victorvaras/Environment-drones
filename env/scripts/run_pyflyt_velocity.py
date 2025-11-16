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
import math
import time
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from env.environment.droneVelocityEnv import DroneVelocityEnv, DroneVelocityEnvConfig  # ajusta si cambias ruta
# Si corres este script directamente desde /mnt/data, puedes hacer:
# from drone_velocity_env import DroneVelocityEnv, DroneVelocityEnvConfig

OUT_DIR = Path.cwd() / "Environment drones" / "salidas_pyflyt-3" 
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    SELECT_MODE = 6 # cambia a 4 o 7 para probar otros

    cfg = DroneVelocityEnvConfig(
        start_xyz=( 0.0, 0.0, 15.0),
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

        #SELECT_MODE = 64

        if SELECT_MODE == 0:
            mov = [0.0, 0.30, 0.0, 1.0]
            dt = 0.1
            for _ in range(100):
                env.step_move(mov, dt)

        elif SELECT_MODE == 1:
            mov = [0.10, 0.0, 0.0, 20.0]
            dt = 0.1
            for _ in range(100):
                env.step_move(mov, dt)

        elif SELECT_MODE == 2:
            mov = [0.010, 0.0, 0.0, 15.0]
            dt = 0.1
            for _ in range(100):
                env.step_move(mov, dt)

        elif SELECT_MODE == 3:
            mov = [0.0, 0.10, 0.0, 15.0]
            dt = 0.1
            for _ in range(100):
                env.step_move(mov, dt)       


        elif SELECT_MODE == 4:
            # Modo 4: [u, v, vr, z]  (u,v en cuerpo; z absoluto en mundo)
            seq4 = [
                ([0.0, 0.0, 0.0, 15.0], 5.0),     # hover en z=5
                ([5.0, 0.0, 0.0, 15.0], 5.0),     # avanzar en u
                ([0.0, 0.0, 0.0, 15.0], 50.0),     # derecha y subir a z=6.5
                ([0.0, 0.0, 5.0, 15.0], 5.0),     # girar en sitio
                ([0.0, 0.0, 0.0, 15.0], 50.0),     # bajar a z=5
                ([5.0, 0.0, 0.0, 15.0], 5.0),
            ]
            # env.step_sequence_mode4(seq4)

            mov = [5.0, 0.0, 0.0, 15.0]
            dt = 0.1
            for _ in range(100):
                env.step_move(mov, dt) 


        elif SELECT_MODE == 5:
            mov = [0.0, 1.0, 0.0, 10.0]
            dt = 0.1
            for _ in range(100):
                env.step_move(mov, dt)       


        elif SELECT_MODE == 6:
            mov = [5.0, 5.0, 0.0, 10.0]
            dt = 0.1
            for _ in range(100):
                env.step_move(mov, dt)       


        
        
        elif SELECT_MODE == 7:
            # Modo 7: [x, y, r, z] (setpoint de posición absoluta en mundo)
            seq7 = [
                ([0.0, 0.0, 0.0, 15.0], 10.0),     # ir a (0,0,15)
                ([25.0, 0.0, 0.0, 15.0], 10.0),     # ir a (5,0,15)
                ([25.0, 25.0, 0.0, 15.0], 10.0),     # ir a (5,5,15) con yaw≈1.2 rad
                ([25.0, 25.0, 0.0, 15.0], 10.0),
                ([0.0, 0.0, 0.0, 15.0], 10.0), 
            ]
            #env.step_sequence_mode7(seq7)


            mov = [20.0, 10.0, 0.0, 15.0] 
            dt = 30.0
            env.step_mode7(mov, dt)

        elif SELECT_MODE == 61:
            # --- Trayectoria circular mínima con un for ---
            z0 = 15.0
            env.step_mode6_holdz(vx=0.0, vy=0.0, vr=0.0, dt=2.0, z_ref=z0)  # estabiliza z

            R = 20.0          # radio [m]
            speed = 5.0       # velocidad tangencial [m/s]
            vueltas = 1.0     # nº de vueltas (puede ser fraccionario)
            dt = 0.1        # duración de cada comando [s]
            follow_heading = True   # True: yaw sigue la tangente; False: yaw fijo
            clockwise = False       # True horario, False antihorario

            # Derivados sencillos
            omega = (speed / R) * (-1.0 if clockwise else 1.0)   # rad/s
            steps = max(1, int((2.0 * math.pi * vueltas) / (abs(omega) * dt)))
            ang = (2.0 * math.pi * vueltas) / steps              # incremento angular por paso

            for i in range(steps):
                theta = i * ang  # ángulo actual

                # Velocidades en marco mundo para trayectoria circular
                vx = -speed * math.sin(theta)
                vy =  speed * math.cos(theta)
                vr = omega if follow_heading else 0.0

                env.step_mode6_holdz(vx=vx, vy=vy, vr=vr, dt=dt, z_ref=z0)

            # Hover breve al terminar
            env.step_mode6_holdz(vx=0.0, vy=0.0, vr=0.0, dt=2.0, z_ref=z0)

        elif SELECT_MODE == 62:
            # Secuencia simple en modo 6: quieto, +X, quieto, +Y, quieto, diagonal (-X,-Y)
            # Usa hold-z para mantener altura constante.
            # "arriba" se interpreta como +Y en el marco del mundo.
            # "abajo a la izquierda" como (-X, -Y) con velocidad diagonal normalizada.

            # Altura objetivo = altura actual
            (x0, y0, z0), _ = env.get_pose()

            speed = 5.0      # m/s
            T = 10.0         # segundos por tramo

            # 1) Quieto 10 s
            #env.step_mode6_holdz(vx=0.0, vy=0.0, vr=0.0, dt=T, z_ref=z0)

            # 2) Avanza en +X 10 s
            env.step_mode6_holdz(vx=speed, vy=0.0, vr=0.0, dt=T, z_ref=z0)

            # 3) Quieto 10 s
            #env.step_mode6_holdz(vx=0.0, vy=0.0, vr=0.0, dt=T, z_ref=z0)

            # 4) Arriba 10 s  (tomado como +Y)
            env.step_mode6_holdz(vx=0.0, vy=speed, vr=0.0, dt=T, z_ref=z0)

            # 5) Quieto 10 s
            #env.step_mode6_holdz(vx=0.0, vy=0.0, vr=0.0, dt=T, z_ref=z0)

            # 6) Abajo a la izquierda 10 s  → (-X, -Y) diagonal
            # Normalizamos para que la magnitud de la velocidad siga siendo 'speed'
            diag = speed / (2 ** 0.5)
            env.step_mode6_holdz(vx=-diag, vy=-diag, vr=0.0, dt=T, z_ref=z0)

            # Hover corto al final (opcional)
            #env.step_mode6_holdz(vx=0.0, vy=0.0, vr=0.0, dt=2.0, z_ref=z0)


        elif SELECT_MODE == 63:
            # Movimiento: X constante; Y oscila en [-2, +2] por ~50 s
            (x0, y0, z0), _ = env.get_pose()

            # Parámetros
            total_time = 50.0   # duración total [s]
            dt = 0.05           # paso de control [s]
            vx_const = 2.0      # velocidad constante en X [m/s]
            A = 2.0             # amplitud de la oscilación de Y [m]  -> Y(t) = y0 + A*sin(ω t)
            period = 5.0        # periodo de oscilación [s] (=> 10 ciclos en 50 s)
            omega = 2.0 * math.pi / period

            # Estabiliza altura al comienzo
            env.step_mode6_holdz(vx=0.0, vy=0.0, vr=0.0, dt=2.0, z_ref=z0)

            steps = int(total_time / dt)
            for i in range(steps):
                t = i * dt
                vx = vx_const
                # Para lograr Y(t) = y0 + A*sin(ω t), usamos vy = dY/dt = A*ω*cos(ω t)
                vy = A * omega * math.cos(omega * t)
                vr = 0.0
                env.step_mode6_holdz(vx=vx, vy=vy, vr=vr, dt=dt, z_ref=z0)

            # Hover corto al final (opcional)
            env.step_mode6_holdz(vx=0.0, vy=0.0, vr=0.0, dt=2.0, z_ref=z0)

        elif SELECT_MODE == 64:
            # X avanza constante; Y toma valores aleatorios dentro de [y0 - A, y0 + A]
            # y se corrige con un controlador proporcional sencillo (no es seno).
            import random

            (x0, y0, z0), _ = env.get_pose()

            # Parámetros
            total_time = 50.0   # duración total [s]
            dt = 2          # paso de control [s]
            vx_const = 5.0      # velocidad constante en X [m/s]
            A = 2.0             # Y debe mantenerse entre y0-A y y0+A
            chunk_dur = 0.1     # cada cuánto tiempo cambiamos el objetivo Y [s]
            k_y = 0.8           # ganancia proporcional para corregir Y
            vy_max = 3.0        # límite de velocidad lateral [m/s]

            # Estabiliza altura al comienzo
            env.step_mode6_holdz(vx=0.0, vy=0.0, vr=0.0, dt=2.0, z_ref=z0)

            # Primer objetivo aleatorio de Y
            y_ref = y0 + random.uniform(-A, A)
            next_switch = chunk_dur

            steps = int(total_time / dt)
            t = 0.0
            for _ in range(steps):
                # Cambio de objetivo aleatorio cada 'chunk_dur' segundos
                if t >= next_switch:
                    y_ref = y0 + random.uniform(-A, A)
                    next_switch += chunk_dur

                # Lee Y actual para calcular el error
                (_, y_cur, _), _rpy = env.get_pose()
                err_y = y_ref - y_cur

                # Control proporcional con saturación
                vy_cmd = k_y * err_y
                if vy_cmd > vy_max:
                    vy_cmd = vy_max
                elif vy_cmd < -vy_max:
                    vy_cmd = -vy_max

                env.step_mode6_holdz(vx=vx_const, vy=vy_cmd, vr=0.0, dt=dt, z_ref=z0)

                t += dt

            # Hover corto al final (opcional)
            env.step_mode6_holdz(vx=0.0, vy=0.0, vr=0.0, dt=2.0, z_ref=z0)




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


"""
¡Sí! PyFlyt (para **QuadX**) trae **9 modos de control**. Cada modo define qué significa el vector de 4 entradas que le mandas al dron. Aquí están, con sus señales y el marco (body vs. mundo) cuando aplica:

**-1. Motores crudos:** `m1, m2, m3, m4`
Control directo de cada motor (útil para tests muy de bajo nivel). ([taijunjet.com][1])

**0. Tasas angulares + empuje:** `vp, vq, vr, T`
`vp,vq,vr`: velocidades angulares (rad/s, marco del cuerpo). `T`: thrust. No compensa gravedad si T es insuficiente. ([taijunjet.com][1])

**1. Ángulos + velocidad vertical:** `p, q, r, vz`
`p,q,r`: ángulos (roll, pitch, yaw) objetivo; `vz`: velocidad vertical en el marco del suelo. ([taijunjet.com][1])

**2. Tasas angulares + altura:** `vp, vq, vr, z`
Controlas la rapidez de giro y una cota de **z** absoluta. ([taijunjet.com][1])

**3. Ángulos + altura:** `p, q, r, z`
Mantiene orientación absoluta y altura objetivo. ([taijunjet.com][1])

**4. Velocidades locales + yaw rate + altura:** `u, v, vr, z`
`u,v` son **velocidades lineales en el marco local (body)**; `vr` tasa de yaw; `z` absoluto. ([taijunjet.com][1])

**5. Velocidades locales + yaw rate + vel. vertical:** `u, v, vr, vz`
`u,v` en **body**; `vz` vertical (suelo). Muy útil si quieres “avanzar hacia donde mira el dron”. ([taijunjet.com][1])

**6. Velocidades en mundo + yaw rate + vel. vertical:** `vx, vy, vr, vz`
`vx,vy,vz` son **velocidades lineales en el marco del suelo (ground)**; `vr` tasa de yaw. Ideal si tu planner trabaja en coordenadas globales. ([taijunjet.com][1])

**7. Posición (x,y,z) + yaw absoluto:** `x, y, r, z`
Waypoints/seguimiento de puntos en mundo. ([taijunjet.com][1])

### Notas rápidas (para que elijas bien)

* **Frames/convenciones:** `u,v,w` = velocidades **locales (body)**; `vx,vy,vz` = velocidades **en mundo (ground)**; `p,q,r` = **ángulos** (roll, pitch, yaw); `vp,vq,vr` = **tasas angulares**; `T` = **thrust**. ([taijunjet.com][1])
* **Lectura de estado:** `Aviary.state(i)` devuelve un `(4,3)` donde `state[3,:]` es **posición en mundo (x,y,z)** y `state[1,:]` son **ángulos (roll,pitch,yaw)**. ([taijunjet.com][2])
* **Controladores internos:** QuadX implementa **PID en cascada**; por eso en modos de velocidad (5/6) el sistema compensa gravedad para cumplir tu `vz`. ([taijunjet.com][1])

### Recomendación para tu proyecto

* Si tu **acción del RL** es “**vector de velocidad + Δt**” en coordenadas globales, usa **modo 6** (`[vx, vy, vr, vz]`).
* Si prefieres mover “hacia adelante del dron” sin importar su yaw, usa **modo 5** (`[u, v, vr, vz]`).
* Deja **modo 0** solo para validaciones de física (hover thrust, gravedad), y **modo 7** para tareas por waypoints.

¿Quieres que te arme una tabla breve comparando pros/cons de 5 vs 6 para integrarlo con tu loop de `step`?

[1]: https://taijunjet.com/PyFlyt/documentation/core/drones/quadx.html "QuadX — PyFlyt"
[2]: https://taijunjet.com/PyFlyt/documentation/core/aviary.html?utm_source=chatgpt.com "Aviary — PyFlyt - Jet"


"""
