# -*- coding: utf-8 -*-
"""
Pequeño wrapper OOP sobre PyFlyt para mover un QuadX con velocidades + Δt.

- Por defecto usa marco del suelo (frame='world') => modo 6: [vx, vy, vr, vz]
- También puedes usar frame='body' => modo 5: [u, v, vr, vz]
- Devuelve posición (x,y,z) y yaw tras cada step(Δt).

Requisitos:
    pip install pyflyt numpy matplotlib
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from typing import Iterable, Tuple, List, Optional

# PyFlyt
from PyFlyt.core import Aviary  # core API del simulador (PyBullet)
# Referencias API:
# - Aviary usage/step/set_mode/set_setpoint/state: https://taijunjet.com/PyFlyt/documentation/core/aviary.html
# - QuadX control modes: https://taijunjet.com/PyFlyt/documentation/core/drones/quadx.html

FrameT = str


@dataclass
class DroneVelocityEnvConfig:
    start_xyz: Tuple[float, float, float] = (0.0, 0.0, 1.0)
    start_rpy: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # en radianes
    control_hz: int = 120
    physics_hz: int = 240
    frame: FrameT = "world"  # 'world' -> modo 6 ; 'body' -> modo 5
    render: bool = False
    drone_model: str = "cf2x"  # QuadX variants: 'cf2x' (default) o 'primitive_drone'
    seed: Optional[int] = None
    record_trajectory: bool = True  # guarda trayecto para graficar luego
    


class DroneVelocityEnv:
    """
    Env minimalista sobre PyFlyt para controlar un QuadX por velocidades.

    API principal:
        - reset() -> (pos, rpy)
        - step(vel4, dt) -> (pos, rpy)   # vel4 = [vx, vy, vr, vz] (world) o [u, v, vr, vz] (body)
        - step_sequence([(vel4, dt), ...]) -> List[(pos, rpy)]
        - get_pose() -> (pos, rpy)
        - close()
        - plot_trajectory()  # opcional, requiere matplotlib
    """

    def __init__(self, cfg: DroneVelocityEnvConfig):
        self.cfg = cfg

        # Validaciones de looprates (control_hz debe dividir a physics_hz)
        if self.cfg.physics_hz % self.cfg.control_hz != 0:
            raise ValueError(
                f"control_hz ({self.cfg.control_hz}) debe dividir a physics_hz "
                f"({self.cfg.physics_hz}) para estabilidad (p.ej., 120 | 240)."
            )

        # Selección de modo según marco:
        # - world => modo 6: [vx, vy, vr, vz] (ground linear velocities + yaw rate)
        # - body  => modo 5: [u, v,  vr, vz] (local linear velocities + yaw rate)
        frame_lower = self.cfg.frame.lower()
        if frame_lower == "world":
            self._mode = 6
        elif frame_lower == "body":
            self._mode = 5
        elif frame_lower == "mode_4":
            self._mode = 4
        else:
            raise ValueError("frame debe ser 'world' o 'body'")

        # Construcción del Aviary
        start_pos = np.array([self.cfg.start_xyz], dtype=float)  # shape (1,3)
        start_orn = np.array([self.cfg.start_rpy], dtype=float)  # shape (1,3)

        drone_options = dict(
            control_hz=self.cfg.control_hz,
            drone_model=self.cfg.drone_model,
        )

        self.env = Aviary(
            start_pos=start_pos,
            start_orn=start_orn,
            render=self.cfg.render,
            drone_type="quadx",
            drone_options=drone_options,
            physics_hz=self.cfg.physics_hz,
            seed=self.cfg.seed,
        )

        # Seteo de modo de control y armado
        self.env.set_mode(self._mode)
        self.env.set_armed(True)

        self.dt_control = 1.0 / float(self.cfg.control_hz)
        self.positions_history: List[np.ndarray] = []
        self.yaw_history: List[float] = []
        self.vel_cmd_history: list[np.ndarray] = []   # [vx, vy, vz, vr]
        self.vel_meas_history: list[np.ndarray] = []  # [vx, vy, vz, vr] (derivadas)


        # Inicialización
        self.reset()

    # -------- API pública --------
    def reset(self) -> Tuple[np.ndarray, np.ndarray]:
        """Reinicia el dron a la pose inicial y devuelve (pos, rpy)."""
        self.env.reset()
        self.env.set_mode(self._mode)
        self.env.set_armed(True)
        pos, rpy = self.get_pose()
        if self.cfg.record_trajectory:
            self.positions_history = [pos.copy()]
            self.yaw_history = [rpy[2]]
            self.vel_cmd_history = []
            self.vel_meas_history = []
        return pos, rpy

    def step(self, vel4: Iterable[float], dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Aplica una velocidad por dt segundos y devuelve (pos, rpy) finales.

        vel4: [vx, vy, vr, vz] en 'world'   (modo 6)
            [u,  v,  vr, vz] en 'body'    (modo 5)
            - vx, vy, u, v, vz en m/s
            - vr en rad/s (yaw rate)

        dt:   segundos de simulación. Se discretiza a pasos de control de 1/control_hz.

        Además, si record_trajectory=True:
        - self.vel_cmd_history acumula [vx, vy, vz, vr] comandado por step
        - self.vel_meas_history acumula [vx, vy, vz, vr] medido (derivado) por step
        """
        v = np.asarray(vel4, dtype=float).reshape(4,)
        if not np.all(np.isfinite(v)):
            raise ValueError("vel4 contiene valores no finitos.")

        # Inicializa historiales si no existen (robustez al copiar/pegar)
        if getattr(self.cfg, "record_trajectory", False):
            if not hasattr(self, "vel_cmd_history"):
                self.vel_cmd_history = []
            if not hasattr(self, "vel_meas_history"):
                self.vel_meas_history = []

        # Cargar setpoint para el dron índice 0 (modo 6: [vx, vy, vr, vz])
        self.env.set_setpoint(0, v)

        # Número de pasos de control a simular
        n_steps = int(round(dt / self.dt_control))
        n_steps = max(n_steps, 1)  # al menos 1 step

        # Valores previos para derivar velocidades medidas en marco mundo
        if getattr(self.cfg, "record_trajectory", False) and getattr(self, "positions_history", None):
            prev_pos = self.positions_history[-1].copy()
            prev_yaw = float(self.yaw_history[-1]) if self.yaw_history else self.get_pose()[1][2]
        else:
            p0, rpy0 = self.get_pose()
            prev_pos = p0.copy()
            prev_yaw = float(rpy0[2])
            # Si no había historial y queremos registrar, arrancamos uno mínimo
            if getattr(self.cfg, "record_trajectory", False) and not getattr(self, "positions_history", None):
                self.positions_history = [prev_pos.copy()]
                self.yaw_history = [prev_yaw]

        pos = prev_pos
        rpy = np.array([0.0, 0.0, prev_yaw], dtype=float)

        for _ in range(n_steps):
            self.env.step()  # un paso de control (maneja subpasos de física internamente)
            pos, rpy = self.get_pose()
            yaw = float(rpy[2])

            if getattr(self.cfg, "record_trajectory", False):
                # 1) Trayectoria e yaw
                self.positions_history.append(pos.copy())
                self.yaw_history.append(yaw)

                # 2) Comando (modo 6): [vx, vy, vr, vz] -> guardo como [vx, vy, vz, vr]
                vx_cmd, vy_cmd, vr_cmd, vz_cmd = float(v[0]), float(v[1]), float(v[2]), float(v[3])
                self.vel_cmd_history.append(np.array([vx_cmd, vy_cmd, vz_cmd, vr_cmd], dtype=float))

                # 3) Medida por derivada en marco mundo
                dp = (pos - prev_pos) / self.dt_control  # [vx, vy, vz] medidos
                # Unwrap de yaw para evitar saltos ±pi
                dyaw = np.unwrap([prev_yaw, yaw])[1] - np.unwrap([prev_yaw, yaw])[0]
                vr_meas = dyaw / self.dt_control
                self.vel_meas_history.append(np.array([dp[0], dp[1], dp[2], vr_meas], dtype=float))

                prev_pos = pos.copy()
                prev_yaw = yaw

        # Pose final tras aplicar dt
        return pos, rpy


    def step_sequence(
        self,
        sequence: Iterable[Tuple[Iterable[float], float]],
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Ejecuta una lista de (vel4, dt). Devuelve lista de (pos, rpy) finales por cada segmento.
        """
        out: List[Tuple[np.ndarray, np.ndarray]] = []
        for vel4, dt in sequence:
            pose = self.step(vel4, dt)
            out.append(pose)
        return out

    def get_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Retorna (pos, rpy) del dron índice 0 desde env.state(0).

        - pos = state[3,:]  -> (x, y, z) en metros (marco mundo)
        - rpy = state[1,:]  -> (roll, pitch, yaw) en radianes
        """
        st = self.env.state(0)  # (4,3)
        pos = np.asarray(st[3, :], dtype=float)  # x,y,z
        rpy = np.asarray(st[1, :], dtype=float)  # roll, pitch, yaw
        return pos, rpy

    def close(self):
        """Cierra el simulador."""
        try:
            self.env.close()
        except Exception:
            pass


    def step_xy_holdz(
        self,
        vx: float, vy: float, vr: float, dt: float,
        z_ref: float | None = None,
        kp: float = 1.6, ki: float = 0.0,
        vz_limit: float = 20.0,
    ):
        """
        Mueve en XY con modo 6 manteniendo la altura cerca de z_ref ajustando vz.
        - vx, vy: velocidades en mundo [m/s]
        - vr: yaw rate [rad/s]
        - dt: duración total del tramo [s]
        - z_ref: altura objetivo; si None, usa la altura actual
        - kp, ki: ganancias para vz = kp*e + ki*∫e (e = z_ref - z)
        - vz_limit: saturación de |vz| [m/s]
        """
        import numpy as np

        # Altura referencia
        if z_ref is None:
            z_ref = float(self.get_pose()[0][2])

        # Integrador
        integ = 0.0

        n_steps = int(round(dt / self.dt_control))
        n_steps = max(1, n_steps)

        # Variables previas para derivadas/registro
        if getattr(self.cfg, "record_trajectory", False) and getattr(self, "positions_history", None):
            prev_pos = self.positions_history[-1].copy()
            prev_yaw = float(self.yaw_history[-1]) if self.yaw_history else self.get_pose()[1][2]
        else:
            p0, rpy0 = self.get_pose()
            prev_pos = p0.copy()
            prev_yaw = float(rpy0[2])
            if getattr(self.cfg, "record_trajectory", False) and not getattr(self, "positions_history", None):
                self.positions_history = [prev_pos.copy()]
                self.yaw_history = [prev_yaw]
                self.vel_cmd_history = []
                self.vel_meas_history = []

        for _ in range(n_steps):
            # Estado actual
            pos, rpy = self.get_pose()
            z = float(pos[2])
            yaw = float(rpy[2])

            # Control P/PI para vz
            e = z_ref - z
            integ += e * self.dt_control
            vz_cmd = kp * e + ki * integ
            vz_cmd = float(np.clip(vz_cmd, -vz_limit, vz_limit))

            # Setpoint modo 6: [vx, vy, vr, vz]
            self.env.set_setpoint(0, np.array([vx, vy, vr, vz_cmd], dtype=float))
            self.env.step()

            # Registro (igual que en step normal)
            if getattr(self.cfg, "record_trajectory", False):
                self.positions_history.append(pos.copy())
                self.yaw_history.append(yaw)

                # Comando guardado como [vx, vy, vz, vr]
                self.vel_cmd_history.append(np.array([vx, vy, vz_cmd, vr], dtype=float))

                dp = (pos - prev_pos) / self.dt_control
                dyaw = np.unwrap([prev_yaw, yaw])[1] - np.unwrap([prev_yaw, yaw])[0]
                vr_meas = dyaw / self.dt_control
                self.vel_meas_history.append(np.array([dp[0], dp[1], dp[2], vr_meas], dtype=float))

                prev_pos = pos.copy()
                prev_yaw = yaw

        # Devuelve pose final
        return self.get_pose()




    # -------- utilidades visuales opcionales --------
    def plot_trajectory(self, show: bool = True, save_path: Optional[str] = None):
        """
        Dibuja la trayectoria acumulada. No usa animación; traza 3D final.
        """
        if not self.positions_history:
            print("[plot_trajectory] No hay datos para graficar.")
            return

        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (necesario para proyección 3d)

        P = np.vstack(self.positions_history)  # (N,3)
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(P[:, 0], P[:, 1], P[:, 2], linewidth=2)
        ax.scatter(P[0, 0], P[0, 1], P[0, 2], s=40, label="inicio")
        ax.scatter(P[-1, 0], P[-1, 1], P[-1, 2], s=40, label="fin")
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.set_zlabel("Z [m]")
        ax.set_title("Trayectoria QuadX (PyFlyt)")
        ax.legend(loc="best")
        ax.grid(True)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)



    def plot_topdown_xy(
        self,
        show: bool = True,
        save_path: str | None = None,
        annotate_heading: bool = True,
        stride: int = 10,
    ):
        """
        Vista superior (X-Y) de la trayectoria.
        - annotate_heading: si True, dibuja flechas en dirección del yaw.
        - stride: dibuja una flecha cada 'stride' muestras (para no saturar).
        """
        if not self.positions_history:
            print("[plot_topdown_xy] No hay datos para graficar.")
            return

        import numpy as np
        import matplotlib.pyplot as plt

        P = np.vstack(self.positions_history)  # (N, 3) -> X,Y,Z
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot(P[:, 0], P[:, 1], linewidth=2)
        ax.scatter(P[0, 0], P[0, 1], s=40, label="inicio")
        ax.scatter(P[-1, 0], P[-1, 1], s=40, label="fin")

        # Opcional: dibujar orientación (yaw) como flechas en XY
        if annotate_heading and len(self.yaw_history) == len(self.positions_history):
            import math
            idx = np.arange(0, len(P), max(1, int(stride)))
            yaws = np.asarray(self.yaw_history)[idx]
            U = np.cos(yaws)  # componente X de la flecha (unidad)
            V = np.sin(yaws)  # componente Y de la flecha (unidad)
            # Escala las flechas para que se vean bien en la figura
            ax.quiver(
                P[idx, 0], P[idx, 1], U, V,
                angles="xy", scale_units="xy", scale=10.0, width=0.003, alpha=0.7
            )

        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.set_title("Trayectoria (vista superior X–Y)")
        ax.grid(True)
        ax.set_aspect("equal", adjustable="datalim")  # relación 1:1 en X-Y

        # Margen visual pequeño alrededor de la trayectoria
        x_min, x_max = np.min(P[:, 0]), np.max(P[:, 0])
        y_min, y_max = np.min(P[:, 1]), np.max(P[:, 1])
        dx = (x_max - x_min) * 0.05 + 1e-6
        dy = (y_max - y_min) * 0.05 + 1e-6
        ax.set_xlim(x_min - dx, x_max + dx)
        ax.set_ylim(y_min - dy, y_max + dy)
        ax.legend(loc="best")

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)


    def plot_velocities(self, compare_cmd: bool = True, show: bool = True, save_path: str | None = None):
        """
        Grafica la evolución temporal de vx, vy, vz y vr (yaw rate).
        - compare_cmd=True: superpone comandado vs medido.
        """
        import numpy as np
        import matplotlib.pyplot as plt

        if not self.vel_meas_history:
            print("[plot_velocities] No hay datos para graficar (ejecuta step primero).")
            return

        V_meas = np.vstack(self.vel_meas_history)  # (N,4) -> [vx,vy,vz,vr]
        N = V_meas.shape[0]
        t = np.arange(N, dtype=float) * self.dt_control

        fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(10, 9), sharex=True)
        labels = [r"$v_x$ [m/s]", r"$v_y$ [m/s]", r"$v_z$ [m/s]", r"$\dot{\psi}$ [rad/s]"]
        for i, ax in enumerate(axes):
            ax.plot(t, V_meas[:, i], linewidth=1.8, label="Medido")
            if compare_cmd and self.vel_cmd_history:
                V_cmd = np.vstack(self.vel_cmd_history)
                M = min(len(V_cmd), N)
                ax.plot(t[:M], V_cmd[:M, i], linestyle="--", linewidth=1.5, label="Comandado")
            ax.set_ylabel(labels[i])
            ax.grid(True)

        axes[-1].set_xlabel("Tiempo [s]")
        axes[0].set_title("Evolución de velocidades (modo 6)")
        axes[0].legend(loc="best")

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)


    def plot_altitude(
        self,
        show: bool = True,
        save_path: str | None = None,
        overlay_vz: bool = False,
        smooth_window: int = 1,
        title: str = "Altura vs tiempo",
    ):
        """
        Grafica la altura z(t) usando self.positions_history.
        Opcional: superpone velocidad vertical medida (derivada de z).

        Params:
            overlay_vz: Si True, dibuja v_z (m/s) en eje secundario.
            smooth_window: Tamaño de ventana (entero >=1) para suavizar v_z con media móvil.
        """
        import numpy as np
        import matplotlib.pyplot as plt

        if not getattr(self, "positions_history", None):
            print("[plot_altitude] No hay trayectoria registrada. Ejecuta step() primero con record_trajectory=True.")
            return

        P = np.vstack(self.positions_history)  # (N,3)
        z = P[:, 2]
        N = len(z)
        t = np.arange(N, dtype=float) * float(self.dt_control)

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(t, z, linewidth=2, label="z (altura) [m]")
        ax.set_xlabel("Tiempo [s]")
        ax.set_ylabel("Altura z [m]")
        ax.set_title(title)
        ax.grid(True)

        if overlay_vz and N >= 2:
            vz = np.diff(z, prepend=z[0]) / float(self.dt_control)

            # Suavizado simple (media móvil) si se pide
            if isinstance(smooth_window, int) and smooth_window > 1:
                k = int(smooth_window)
                kernel = np.ones(k, dtype=float) / float(k)
                vz = np.convolve(vz, kernel, mode="same")

            ax2 = ax.twinx()
            ax2.plot(t, vz, linestyle="--", linewidth=1.5, alpha=0.7, label="v_z (medida) [m/s]")
            ax2.set_ylabel("v_z [m/s]")

            # Leyenda combinada
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc="best")
        else:
            ax.legend(loc="best")

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)


    # ------ MODO 4 ------

    # --- Helpers de modo ---
    def enter_mode4(self):
        """Cambia a modo 4 (u, v, vr, z) y arma el dron."""
        self.env.set_mode(4)
        self.env.set_armed(True)

    def restore_default_mode(self):
        """Vuelve al modo por defecto configurado en __init__ (5 o 6)."""
        self.env.set_mode(self._mode)
        self.env.set_armed(True)

    def step_mode4(self, uvrz: Iterable[float], dt: float, restore_mode: bool = True):
        """
        Modo 4 DIRECTO: aplica [u, v, vr, z] durante dt segundos y devuelve (pos, rpy).

        Parámetros:
            uvrz: iterable de 4 floats -> [u, v, vr, z]
                - u, v : velocidades lineales en MARCO DEL CUERPO [m/s]
                        u = hacia delante del dron, v = hacia la derecha del dron
                - vr   : velocidad angular de yaw [rad/s]
                - z    : ALTURA absoluta objetivo (en marco mundo) [m]
            dt   : duración del tramo [s]
            restore_mode: si True, vuelve al modo original (p. ej., 6) al terminar.

        Registro:
            - Guarda pose/yaw como siempre.
            - En vel_cmd_history guarda [u, v, NaN, vr] (vz no existe en modo 4).
            - vel_meas_history sigue siendo [vx, vy, vz, vr] medidos (derivadas en mundo).
        """
        import numpy as np

        sp = np.asarray(uvrz, dtype=float).reshape(4,)
        if not np.all(np.isfinite(sp)):
            raise ValueError("uvrz contiene valores no finitos.")
        # Cambiamos a modo 4
        current_mode = getattr(self, "_mode", 6)
        self.enter_mode4()

        n_steps = max(1, int(round(dt / self.dt_control)))

        # Previos para derivadas/registro
        if getattr(self.cfg, "record_trajectory", False) and getattr(self, "positions_history", None):
            prev_pos = self.positions_history[-1].copy()
            prev_yaw = float(self.yaw_history[-1]) if self.yaw_history else self.get_pose()[1][2]
        else:
            p0, rpy0 = self.get_pose()
            prev_pos = p0.copy()
            prev_yaw = float(rpy0[2])
            if getattr(self.cfg, "record_trajectory", False) and not getattr(self, "positions_history", None):
                self.positions_history = [prev_pos.copy()]
                self.yaw_history = [prev_yaw]
                self.vel_cmd_history = []
                self.vel_meas_history = []

        pos, rpy = prev_pos, np.array([0.0, 0.0, prev_yaw], dtype=float)

        for _ in range(n_steps):
            # Setpoint nativo de modo 4: [u, v, vr, z]
            self.env.set_setpoint(0, sp)
            self.env.step()

            pos, rpy = self.get_pose()
            yaw = float(rpy[2])

            if getattr(self.cfg, "record_trajectory", False):
                # Pose/yaw
                self.positions_history.append(pos.copy())
                self.yaw_history.append(yaw)

                # Comando guardado con mismo shape [vx, vy, vz, vr]; aquí vz = NaN
                self.vel_cmd_history.append(np.array([sp[0], sp[1], np.nan, sp[2]], dtype=float))

                # Medidas (en mundo) por derivada
                dp = (pos - prev_pos) / self.dt_control
                dyaw = np.unwrap([prev_yaw, yaw])[1] - np.unwrap([prev_yaw, yaw])[0]
                vr_meas = dyaw / self.dt_control
                self.vel_meas_history.append(np.array([dp[0], dp[1], dp[2], vr_meas], dtype=float))

                prev_pos = pos.copy()
                prev_yaw = yaw

        # Restaurar modo original si se pide
        if restore_mode:
            self.env.set_mode(current_mode)
            self.env.set_armed(True)

        return pos, rpy


    def step_sequence_mode4(self, sequence: Iterable[Tuple[Iterable[float], float]], restore_mode_each: bool = False):
        """
        Ejecuta una lista de (uvrz, dt) en modo 4.
        Si restore_mode_each=False, permanece en modo 4 durante toda la secuencia.
        """
        poses = []
        stayed_in_mode4 = False
        if not restore_mode_each:
            self.enter_mode4()
            stayed_in_mode4 = True
        for sp, dur in sequence:
            poses.append(self.step_mode4(sp, dur, restore_mode=False if stayed_in_mode4 else True))
        if stayed_in_mode4:
            self.restore_default_mode()
        return poses




# modo  0

    def _set_mode(self, mode: int):
        """Cambia modo del QuadX preservando armado."""
        self.env.set_mode(mode)
        self.env.set_armed(True)

    def enter_mode0(self):
        """Pasa a modo 0: [vp, vq, vr, T]."""
        self._set_mode(0)

    def restore_default_mode(self):
        """Vuelve al modo configurado por frame ('world'->6, 'body'->5)."""
        self._set_mode(self._mode)

    def step_mode0(self, setpoint4, dt: float):
        """
        Un paso en modo 0 con setpoint4 = [vp, vq, vr, T].
        vp,vq,vr en rad/s; T = thrust (unidad interna de PyFlyt).
        """
        s = np.asarray(setpoint4, dtype=float).reshape(4,)
        if not np.all(np.isfinite(s)):
            raise ValueError("setpoint4 contiene valores no finitos.")

        self.env.set_setpoint(0, s)

        n_steps = int(round(dt / self.dt_control))
        n_steps = max(n_steps, 1)
        for _ in range(n_steps):
            self.env.step()
            if self.cfg.record_trajectory:
                p, rpy = self.get_pose()
                self.positions_history.append(p.copy())
                self.yaw_history.append(rpy[2])

        return self.get_pose()

    def plot_altitude_profile(self, show: bool = True, save_path: str | None = None):
        """Grafica z(t) para verificar caída/subida/hover."""
        if not self.positions_history:
            print("[plot_altitude_profile] No hay trayectoria.")
            return
        import numpy as np, matplotlib.pyplot as plt
        P = np.vstack(self.positions_history)
        N = len(P)
        t = np.arange(N) * self.dt_control
        fig, ax = plt.subplots(figsize=(7, 3.6))
        ax.plot(t, P[:, 2], linewidth=2)
        ax.set_xlabel("Tiempo [s]")
        ax.set_ylabel("Altura z [m]")
        ax.set_title("Perfil de altura")
        ax.grid(True)
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)

    def find_hover_thrust_mode0(self, t_seg: float = 1.5, t_warmup: float = 0.3,
                                T_init_grid: list[float] | None = None,
                                max_iter: int = 10, tol_dz_dt: float = 0.02):
        """
        Estima un thrust ~hover en modo 0 por búsqueda (signo de velocidad vertical).
        - t_seg: duración por intento (s)
        - t_warmup: porción inicial descartada para estimar dz/dt
        - T_init_grid: lista inicial de T para sondear rango (auto si None)
        - tol_dz_dt: umbral de |dz/dt| (m/s) para considerar hover
        Devuelve (T_hover, hist) donde hist=[(T, dz_dt_prom)].
        """
        import numpy as np

        # Nos aseguramos de estar en modo 0 y empezamos desde la pose actual
        self.enter_mode0()

        # Grid inicial robusto (funciona si T es normalizado ~[0..1] o más grande)
        if T_init_grid is None:
            T_init_grid = [0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0]

        hist = []

        def measure_dzdt(T):
            # Cero rates angulares para que el efecto sea puramente vertical
            self.step_mode0([0.0, 0.0, 0.0, T], t_seg)
            # Estimamos dz/dt en la cola de la ventana (evitar calentamiento)
            samples = int(round(t_warmup / self.dt_control))
            P = np.vstack(self.positions_history)
            z = P[-int(round(t_seg / self.dt_control)):, 2]
            if samples < len(z):
                z = z[samples:]
            # Regresión lineal simple para dz/dt
            t_local = np.arange(len(z)) * self.dt_control
            if len(z) < 2:
                return 0.0
            A = np.vstack([t_local, np.ones_like(t_local)]).T
            slope, _ = np.linalg.lstsq(A, z, rcond=None)[0]
            return float(slope)

        # 1) Barrido para encontrar un rango [T_lo, T_hi] que cruce el cero de dz/dt
        T_lo, v_lo = None, None
        T_hi, v_hi = None, None
        for T in T_init_grid:
            v = measure_dzdt(T)
            hist.append((T, v))
            if abs(v) <= tol_dz_dt:
                return T, hist
            if v < 0 and (T_lo is None or T > T_lo):
                T_lo, v_lo = T, v
            if v > 0 and (T_hi is None or T < T_hi):
                T_hi, v_hi = T, v

        # Si no se encontró cruce, ampliamos por si T estaba mal escalado
        if T_lo is None or T_hi is None:
            expand = 1.0
            for _ in range(4):
                expand *= 2.0
                T_candidates = []
                if T_lo is None:
                    T_candidates.append(0.0)
                T_candidates.append(expand)
                for T in T_candidates:
                    v = measure_dzdt(T)
                    hist.append((T, v))
                    if v < 0 and (T_lo is None or T > T_lo):
                        T_lo, v_lo = T, v
                    if v > 0 and (T_hi is None or T < T_hi):
                        T_hi, v_hi = T, v
                    if T_lo is not None and T_hi is not None:
                        break
                if T_lo is not None and T_hi is not None:
                    break

        if T_lo is None or T_hi is None:
            print("[WARN] No se pudo acotar T_hover; revisa escala de thrust.")
            return None, hist

        # 2) Búsqueda binaria en [T_lo, T_hi]
        for _ in range(max_iter):
            T_mid = 0.5 * (T_lo + T_hi)
            v_mid = measure_dzdt(T_mid)
            hist.append((T_mid, v_mid))
            if abs(v_mid) <= tol_dz_dt:
                return T_mid, hist
            if v_mid > 0:  # sube -> exceso de thrust
                T_hi, v_hi = T_mid, v_mid
            else:          # baja -> faltante de thrust
                T_lo, v_lo = T_mid, v_mid

        T_est = 0.5 * (T_lo + T_hi)
        return T_est, hist

