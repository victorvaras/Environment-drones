# -*- coding: utf-8 -*-
"""
Pequeña demo para validar el wrapper con control por velocidades.

Ejecuta varios segmentos (velocidad, duración) y muestra la trayectoria.
- Con render=True verás la ventana de PyBullet.
- Cambia frame='world' por 'body' si prefieres velocidades en marco local.

Ejemplo de integración a tu Gym:
    # obtienes la nueva posición tras cada step y la pasas a tu RT, etc.
    pos, rpy = env_pyflyt.step([vx, vy, vr, vz], dt)
"""
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import math
from env.environment.drone_velocity_env import DroneVelocityEnv, DroneVelocityEnvConfig
import time

def main():
    cfg = DroneVelocityEnvConfig(
        start_xyz=(0.0, 0.0, 10.0),
        start_rpy=(0.0, 0.0, 0.0),
        control_hz=120,
        physics_hz=240,
        frame="mode_4",         # 'world' => [vx, vy, vr, vz]; 'body' => [u, v, vr, vz]
        render=False,           # True para ver el movimiento ahora; False para headless
        drone_model="cf2x",
        seed=42,
        record_trajectory=True,
    )

    env = DroneVelocityEnv(cfg)
    try:
        print("Pose inicial:", env.get_pose())

        # --- Mantener altura z≈1.0 m mientras se mueve en XY ---
        # Usa el bloqueo de altura con PI simple (aquí sólo P: kp=1.2, ki=0.0).
        z0 = 10

        # estatico manteniendo z=z0
        #env.step_xy_holdz(vx=0, vy=0.0, vr=0.0, dt=10.0, z_ref=z0, kp=1.2, ki=0.0)

        # Recta en X 10 s manteniendo z=z0
        #env.step_xy_holdz(vx=1.0, vy=0.0, vr=0.0, dt=10.0, z_ref=z0, kp=1.2, ki=0.0)
        # Recta en Y 6 s manteniendo z=z0
        #env.step_xy_holdz(vx=0.0, vy=1.0, vr=0.0, dt=10.0,  z_ref=z0, kp=1.2, ki=0.0)
        # Giro en sitio 6 s manteniendo z=z0
        #env.step_xy_holdz(vx=0.0, vy=0.0, vr=0.4, dt=6.0,  z_ref=z0, kp=1.2, ki=0.0)
        # Hover 4 s (vx=vy=vr=0) manteniendo z=z0
        #env.step_xy_holdz(vx=0.0, vy=0.0, vr=0.0, dt=10.0,  z_ref=z0, kp=1.2, ki=0.0)


        """
        pulsos_xy_vr = [
            (+1.0,  0.0,  0.0),  # +X
            ( 0.0, +1.0,  0.0),  # +Y
            (-1.0,  0.0,  0.0),  # -X
            ( 0.0, -1.0,  0.0),  # -Y

            (+0.8, +0.8,  0.0),  # diag +X+Y
            (-0.8, +0.8,  0.0),  # diag -X+Y
            (+0.8, -0.8,  0.0),  # diag +X-Y
            (-0.8, -0.8,  0.0),  # diag -X-Y

            ( 0.0,  0.0, +0.8),  # yaw CW
            ( 0.0,  0.0, -0.8),  # yaw CCW

            ( 0.0,  0.0,  0.0),  # hover breve
        ]
        """
        pulsos_xy_vr = [
            (0,  0.0,  0.0),
            (5.0,  0.0,  0.0),  
            #(0,  0.0,  0.0),
            (0.0, 0.0,  0.0),
            #(0.0, 0.0, 0.0),
            #(0,0,0),
            #(0,0,0),
        ]

        tiempo_inicio = time.perf_counter()
        for vx, vy, vr in pulsos_xy_vr:
            for i in range(30):
                #env.step_xy_holdz(vx=vx, vy=vy, vr=vr, dt=0.2, z_ref=z0, kp=1.6, ki=0.1)
                i= i+1

        tiempo_total = time.perf_counter() - tiempo_inicio
        print(f"Tiempo total de movimientos: {tiempo_total:.2f} s")

        S4 = [
            ([0,  0.0, 0.0, z0], 40.0),   # u +1 (adelante del dron) 2 s
            ([ 5.0, 0.0, 0.0, z0], 40.0),   # v +1 (derecha del dron)   2 s
            ([ 0.0,  0.0, 0.0, z0], 10.0),   # yaw rate +0.5 rad/s       2 s
            ([ 0.0,  5.0, 0.0, z0], 10.0),   # hover a z0
            ([ 0.0,  0.0, 0.0, z0], 10.0),   # hover a z0
            ([ 0.0,  0.0, 0.0, z0], 10.0),   # hover a z0
        ]
        env.step_sequence_mode4(S4)  # mantiene modo 4 dentro y lo restaura al final


        # --- Pose final ---
        print("Pose final:", env.get_pose())

        # --- Gráficas ---
        # (1) Trayectoria 3D opcional
        env.plot_trajectory(show=True, save_path=None)

        # (2) Vista superior X-Y (con flechas de orientación cada 12 muestras)
        env.plot_topdown_xy(show=True, save_path=None, annotate_heading=True, stride=12)

        # (3) Velocidades comandadas vs medidas
        env.plot_velocities(compare_cmd=True, show=True, save_path=None)

        # (4) Altura z(t) y opcionalmente v_z(t) (usa alias para tu llamada actual)
        env.plot_altitude_profile(show=True, save_path=None)
        env.plot_altitude_profile(show=True, overlay_vz=True, smooth_window=5, save_path=None)

    finally:
        print("Posición final (al cerrar):", env.get_pose())
        env.close()


if __name__ == "__main__":
    main()

    #demo_mode0_gravity() 



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
