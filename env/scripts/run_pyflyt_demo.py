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

OUT_DIR = Path.cwd() / "Environment drones" / "salidas_pyflyt_2"
OUT_DIR.mkdir(parents=True, exist_ok=True)


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
    SELECT_MODE = 6

    env = DroneVelocityEnv(cfg)
    try:
        print("Pose inicial:", env.get_pose())

        if SELECT_MODE == 6:
            # --- Trayectoria circular mínima con un for ---
            z0 = 15.0
            env.step_mode6_holdz(vx=0.0, vy=0.0, vr=0.0, dt=2.0, z_ref=z0)  # estabiliza z

            R = 20.0          # radio [m]
            speed = 5.0       # velocidad tangencial [m/s]
            vueltas = 1.0     # nº de vueltas (puede ser fraccionario)
            dt = 0.05         # duración de cada comando [s]
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




        # --- Pose final ---
        print("Pose final:", env.get_pose())

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
