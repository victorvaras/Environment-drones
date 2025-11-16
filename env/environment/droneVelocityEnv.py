"""
DroneVelocityEnv — Wrapper minimalista y adaptable para PyFlyt (QuadX)
con soporte explícito para los modos 4, 6 y 7 en un *único* init.

Basado en la documentación oficial de PyFlyt (QuadX.set_mode) y su paper.
Requisitos:
    pip install pyflyt numpy matplotlib

Modos soportados (según PyFlyt QuadX.set_mode):
    4: [u, v,  vr, z]   -> velocidades *en el cuerpo* + altura absoluta (mundo)
    6: [vx, vy, vr, vz] -> velocidades *en el mundo* + velocidad vertical
    7: [x,  y,  r,  z]  -> posición (mundo) + yaw absoluto

Esta clase permite:
  - Inicializar el dron indicando directamente el modo (4, 6 ó 7).
  - Ejecutar pasos con los *setpoints nativos* de cada modo (sin transformaciones intermedias).
  - (Modo 6) Opción auxiliar para "auto-ajustar" altura a un z_objetivo (calculando vz).
  - Registrar trayectoria y graficar 3D, XY, velocidades y perfil de altura.

Autor: ChatGPT (para Víctor)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple, List, Optional

import numpy as np

# PyFlyt
from PyFlyt.core import Aviary  # Motor del simulador (PyBullet)


# ----------------------------- Configuración ---------------------------------

@dataclass
class DroneVelocityEnvConfig:
    # Pose inicial
    start_xyz: Tuple[float, float, float] = (0.0, 0.0, 1.0)
    start_rpy: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # [roll, pitch, yaw] en rad

    # Frecuencias
    control_hz: int = 120
    physics_hz: int = 240

    # Modo de control (OBLIGATORIO: 4, 6 o 7)
    mode: int = 6

    # Render
    render: bool = False

    # Modelo de dron: 'cf2x' (Crazyflie 2.x) o 'primitive_drone'
    drone_model: str = "cf2x"

    # Semilla y registro
    seed: Optional[int] = None
    record_trajectory: bool = True


# ----------------------------- Clase principal --------------------------------

class DroneVelocityEnv:
    """
    Wrapper minimalista para controlar un QuadX de PyFlyt.

    API:
        - reset() -> (pos, rpy)
        - close()
        - set_mode(mode)
        - get_pose() -> (pos, rpy)

        - step_mode4([u, v, vr, z], dt)
        - step_mode6([vx, vy, vr, vz], dt)
        - step_mode6_holdz(vx, vy, vr, dt, z_ref, kp=..., ki=..., vz_limit=...)
        - step_mode7([x, y, r, z], dt)

        - step_sequence_modeX(lista_de_(sp4, dt))  # para 4/6/7

        - plot_trajectory_3d(...)
        - plot_xy(...)
        - plot_velocities(...)
        - plot_altitude(...)
    """

    # --------------------------- construcción ---------------------------------

    def __init__(self, cfg: DroneVelocityEnvConfig,
                 step_durations: float = 0.1
                 ):
        self.cfg = cfg


        # Construcción del Aviary
        start_pos = np.array([self.cfg.start_xyz], dtype=float)  # (1,3)
        start_orn = np.array([self.cfg.start_rpy], dtype=float)  # (1,3)
        drone_options = dict(control_hz=self.cfg.control_hz, drone_model=self.cfg.drone_model)

        self.env = Aviary(
            start_pos=start_pos,
            start_orn=start_orn,
            render=self.cfg.render,
            drone_type="quadx",
            drone_options=drone_options,
            physics_hz=self.cfg.physics_hz,
            seed=self.cfg.seed,
        )

        self.dt_control = 1.0 / float(self.cfg.control_hz)

        # Historiales
        self.positions_history: List[np.ndarray] = []
        self.yaw_history: List[float] = []
        self.cmd_history: List[np.ndarray] = []   # [vx, vy, vz, vr] o equivalente según modo
        self.meas_history: List[np.ndarray] = []  # [vx, vy, vz, vr] medidos (mundo)

        self.step_durations = step_durations

        # Estado
        self.default_mode = int(self.cfg.mode)
        self.reset()

    # ---------------------------- utilidades base ------------------------------

    def reset(self):
        self.env.reset()
        self.set_mode(self.default_mode)
        self.env.set_armed(True)

        pos, rpy = self.get_pose()
        if self.cfg.record_trajectory:
            self.positions_history = [pos.copy()]
            self.yaw_history = [float(rpy[2])]
            self.cmd_history = []
            self.meas_history = []
        return pos, rpy

    def close(self):
        try:
            self.env.close()
        except Exception:
            pass

    def set_mode(self, mode: int):
        if mode not in (-1, 0, 1, 2, 3, 4, 5, 6, 7):
            raise ValueError("Modo inválido para QuadX.")
        self.env.set_mode(int(mode))

    def get_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Retorna (pos, rpy) del dron índice 0:
          - pos = [x, y, z] en *mundo*
          - rpy = [roll, pitch, yaw] en rad
        """
        st = self.env.state(0)  # shape (4,3)
        pos = np.asarray(st[3, :], dtype=float)
        rpy = np.asarray(st[1, :], dtype=float)
        return pos, rpy

    # --------------------------- helpers internos ------------------------------

    def _run_for(self, dt: float, setpoint: np.ndarray, log_as: str):
        """
        Ejecuta 'dt' segundos aplicando 'setpoint' en cada paso de control.
        - 'log_as' indica cómo mapear el comando a [vx, vy, vz, vr] para graficar.
        """
        sp = np.asarray(setpoint, dtype=float).reshape(4,)
        if not np.all(np.isfinite(sp)):
            raise ValueError("Setpoint contiene valores no finitos.")
        n_steps = max(1, int(round(dt / self.dt_control)))

        # Previos para derivadas
        pos_prev, rpy_prev = self.get_pose()
        yaw_prev = float(rpy_prev[2])

        for _ in range(n_steps):
            self.env.set_setpoint(0, sp)
            self.env.step()

            pos, rpy = self.get_pose()
            yaw = float(rpy[2])

            if self.cfg.record_trajectory:
                # 1) trayectoria
                self.positions_history.append(pos.copy())
                self.yaw_history.append(yaw)

                # 2) comando -> lo registramos en formato [vx, vy, vz, vr] para comparar
                if log_as == "mode6":
                    vx_cmd, vy_cmd, vr_cmd, vz_cmd = sp[0], sp[1], sp[2], sp[3]
                    self.cmd_history.append(np.array([vx_cmd, vy_cmd, vz_cmd, vr_cmd], dtype=float))
                elif log_as == "mode4":
                    # [u, v, vr, z] -> no hay vz comandado explícitamente
                    self.cmd_history.append(np.array([np.nan, np.nan, np.nan, sp[2]], dtype=float))
                elif log_as == "mode7":
                    # [x, y, r, z] -> comando de posición; no hay velocidades comandadas
                    self.cmd_history.append(np.array([np.nan, np.nan, np.nan, np.nan], dtype=float))
                else:
                    self.cmd_history.append(np.array([np.nan, np.nan, np.nan, np.nan], dtype=float))

                # 3) medición (mundo) por derivada
                dp = (pos - pos_prev) / self.dt_control
                # Unwrap yaw para evitar saltos
                dyaw = np.unwrap([yaw_prev, yaw])[1] - np.unwrap([yaw_prev, yaw])[0]
                vr_meas = dyaw / self.dt_control
                self.meas_history.append(np.array([dp[0], dp[1], dp[2], vr_meas], dtype=float))

            pos_prev = pos
            yaw_prev = yaw

        return pos, rpy

    # Aplicacion de movimiento, para cualquier modo (interno)
    def step_move(self, move: Iterable[float], dt: float):
        
        sp = np.asarray(move, dtype=float).reshape(4,)

        pos, rpy = self._run_for(dt, sp, log_as="generic")
        return pos

    # ----------------------- MODO 4: [u, v, vr, z] -----------------------------

    def step_mode4(self, uvrz: Iterable[float], dt: float):
        """Aplica Modo 4 DIRECTO durante dt segundos y devuelve (pos, rpy)."""
        self.set_mode(4)
        return self._run_for(dt, np.asarray(uvrz, dtype=float), log_as="mode4")

    def step_sequence_mode4(self, seq: Iterable[Tuple[Iterable[float], float]]):
        out: List[Tuple[np.ndarray, np.ndarray]] = []
        self.set_mode(4)
        for sp4, dt in seq:
            out.append(self._run_for(dt, np.asarray(sp4, dtype=float), log_as="mode4"))
        return out

    # ----------------------- MODO 6: [vx, vy, vr, vz] -------------------------

    def step_mode6(self, vxvyvrvz: Iterable[float], dt: float):
        """Aplica Modo 6 DIRECTO durante dt segundos y devuelve (pos, rpy)."""
        self.set_mode(6)
        return self._run_for(dt, np.asarray(vxvyvrvz, dtype=float), log_as="mode6")

    def step_sequence_mode6(self, seq: Iterable[Tuple[Iterable[float], float]]):
        out: List[Tuple[np.ndarray, np.ndarray]] = []
        self.set_mode(6)
        for sp4, dt in seq:
            out.append(self._run_for(dt, np.asarray(sp4, dtype=float), log_as="mode6"))
        return out

    def step_mode6_holdz(
        self,
        vx: float, vy: float, vr: float, dt: float,
        z_ref: Optional[float] = None,
        kp: float = 2.0, ki: float = 0.0,
        vz_limit: float = 8.0,
    ):
        """
        Auxiliar para Modo 6: mantiene/ajusta altura a z_ref "lo antes posible".
        Calcula vz = sat(kp*e + ki*∫e), e = (z_ref - z).
        """
        self.set_mode(6)

        # Altura referencia
        if z_ref is None:
            z_ref = float(self.get_pose()[0][2])

        integ = 0.0
        n_steps = max(1, int(round(dt / self.dt_control)))

        pos_prev, rpy_prev = self.get_pose()
        yaw_prev = float(rpy_prev[2])

        for _ in range(n_steps):
            # Estado actual
            pos, rpy = self.get_pose()
            z = float(pos[2])
            yaw = float(rpy[2])

            # Control PI en z -> vz
            e = z_ref - z
            integ += e * self.dt_control
            vz_cmd = float(np.clip(kp * e + ki * integ, -vz_limit, vz_limit))

            sp = np.array([vx, vy, vr, vz_cmd], dtype=float)
            self.env.set_setpoint(0, sp)
            self.env.step()

            # Log
            if self.cfg.record_trajectory:
                self.positions_history.append(pos.copy())
                self.yaw_history.append(yaw)

                self.cmd_history.append(np.array([vx, vy, vz_cmd, vr], dtype=float))

                dp = (pos - pos_prev) / self.dt_control
                dyaw = np.unwrap([yaw_prev, yaw])[1] - np.unwrap([yaw_prev, yaw])[0]
                vr_meas = dyaw / self.dt_control
                self.meas_history.append(np.array([dp[0], dp[1], dp[2], vr_meas], dtype=float))

            pos_prev = pos
            yaw_prev = yaw

        return self.get_pose()

    # ----------------------- MODO 7: [x, y, r, z] -----------------------------

    def step_mode7(self, xyrz: Iterable[float], dt: float):
        """Aplica Modo 7 DIRECTO durante dt segundos y devuelve (pos, rpy)."""
        self.set_mode(7)
        return self._run_for(dt, np.asarray(xyrz, dtype=float), log_as="mode7")

    def step_sequence_mode7(self, seq: Iterable[Tuple[Iterable[float], float]]):
        out: List[Tuple[np.ndarray, np.ndarray]] = []
        self.set_mode(7)
        for sp4, dt in seq:
            out.append(self._run_for(dt, np.asarray(sp4, dtype=float), log_as="mode7"))
        return out

    # ------------------------------- Gráficas ---------------------------------

    def plot_trajectory_3d(self, show: bool = True, save_path: Optional[str] = None):
        if not self.positions_history:
            print("[plot_trajectory_3d] No hay datos para graficar.")
            return
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        P = np.vstack(self.positions_history)
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(P[:, 0], P[:, 1], P[:, 2], linewidth=2)
        ax.scatter(P[0, 0], P[0, 1], P[0, 2], s=40, label="inicio")
        ax.scatter(P[-1, 0], P[-1, 1], P[-1, 2], s=40, label="fin")
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.set_zlabel("Z [m]")
        ax.set_title("Trayectoria 3D (PyFlyt QuadX)")
        ax.legend(loc="best")
        ax.grid(True)
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)

    def plot_xy(self, show: bool = True, save_path: Optional[str] = None, annotate_heading: bool = True, stride: int = 12):
        if not self.positions_history:
            print("[plot_xy] No hay datos para graficar.")
            return
        import matplotlib.pyplot as plt
        import numpy as np

        P = np.vstack(self.positions_history)
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot(P[:, 0], P[:, 1], linewidth=2)
        ax.scatter(P[0, 0], P[0, 1], s=40, label="inicio")
        ax.scatter(P[-1, 0], P[-1, 1], s=40, label="fin")

        if annotate_heading and len(self.yaw_history) == len(self.positions_history):
            
            idx = np.arange(0, len(P), max(1, int(stride)))
            yaws = np.asarray(self.yaw_history)[idx]
            U = np.cos(yaws)
            V = np.sin(yaws)
            ax.quiver(P[idx, 0], P[idx, 1], U, V, angles="xy", scale_units="xy", scale=10.0, width=0.003, alpha=0.7)

        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.set_title("Trayectoria (vista superior X–Y)")
        ax.grid(True)
        ax.legend(loc="best")
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)

    def plot_velocities(self, compare_cmd: bool = True, show: bool = True, save_path: Optional[str] = None, title: str | None = None):
        if not self.meas_history:
            print("[plot_velocities] No hay datos para graficar (ejecuta algún step primero).")
            return
        import numpy as np
        import matplotlib.pyplot as plt

        V_meas = np.vstack(self.meas_history)  # [vx, vy, vz, vr] medidos
        N = V_meas.shape[0]
        t = np.arange(N, dtype=float) * self.dt_control
        labels = [r"$v_x$ [m/s]", r"$v_y$ [m/s]", r"$v_z$ [m/s]", r"$\dot{\psi}$ [rad/s]"]

        fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(10, 9), sharex=True)
        for i, ax in enumerate(axes):
            ax.plot(t, V_meas[:, i], linewidth=1.8, label="Medido")
            if compare_cmd and self.cmd_history:
                V_cmd = np.vstack(self.cmd_history)
                M = min(len(V_cmd), N)
                ax.plot(t[:M], V_cmd[:M, i], linestyle="--", linewidth=1.5, label="Comandado")
            ax.set_ylabel(labels[i])
            ax.grid(True)
        axes[-1].set_xlabel("Tiempo [s]")
        axes[0].set_title(title or "Evolución de velocidades (mundo)")
        axes[0].legend(loc="best")
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)

    def plot_altitude(self, show: bool = True, save_path: Optional[str] = None, overlay_vz: bool = False, smooth_window: int = 1, title: str = "Altura vs tiempo"):
        if not self.positions_history:
            print("[plot_altitude] No hay trayectoria registrada.")
            return
        import numpy as np
        import matplotlib.pyplot as plt

        P = np.vstack(self.positions_history)
        z = P[:, 2]
        N = len(z)
        t = np.arange(N, dtype=float) * self.dt_control

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(t, z, linewidth=2, label="z (altura) [m]")
        ax.set_xlabel("Tiempo [s]")
        ax.set_ylabel("Altura z [m]")
        ax.set_title(title)
        ax.grid(True)

        if overlay_vz and N >= 2:
            vz = np.diff(z, prepend=z[0]) / float(self.dt_control)
            # suavizado simple si se pide
            if isinstance(smooth_window, int) and smooth_window > 1:
                k = int(smooth_window)
                kernel = np.ones(k, dtype=float) / float(k)
                vz = np.convolve(vz, kernel, mode="same")

            ax2 = ax.twinx()
            ax2.plot(t, vz, linestyle="--", linewidth=1.5, alpha=0.7, label="v_z [m/s]")
            ax2.set_ylabel("v_z [m/s]")
            # leyenda combinada
            L1, l1 = ax.get_legend_handles_labels()
            L2, l2 = ax2.get_legend_handles_labels()
            ax.legend(L1 + L2, l1 + l2, loc="best")
        else:
            ax.legend(loc="best")

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)


    def _infer_step_params(self):
        """
        Devuelve (ctrls_per_step, dt_step).
        - Si existe self.ctrls_per_step lo usa directamente.
        - Si no, intenta inferir con self.rl_hz (si existe).
        - Si no hay nada, asume 1 control por step (equivalente a step = índice).
        """
        dtc = float(self.dt_control)
        control_hz = 1.0 / dtc

        # Opción 1: atributo explícito
        cps = getattr(self, "ctrls_per_step", None)
        if isinstance(cps, (int, float)) and cps >= 1:
            cps = int(round(cps))
        else:
            # Opción 2: inferir desde rl_hz si existe
            rl_hz = getattr(self, "rl_hz", None)
            if rl_hz is not None and rl_hz > 0:
                cps = max(1, int(round(control_hz / float(rl_hz))))
            else:
                cps = 1

        dt_step = cps * dtc
        return cps, dt_step





    def save_all_plots(
        self,
        out_dir,
        name: str = "run",
        annotate_heading: bool = True,
        stride: int = 12,
        include_vz: bool = True,
        smooth_window: int = 5,
    ) -> dict:
        """
        Guarda todas las gráficas en 'out_dir' con prefijo 'name' y NO abre ventanas.
        Retorna un dict con las rutas de los PNGs.
        """
        from pathlib import Path
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        files = {}
        files["traj3d"] = str(out_path / f"Trayectoria 3D.png")
        files["xy"]     = str(out_path / f"Trayectoria XY.png")
        files["vel"]    = str(out_path / f"Velocidades.png")
        files["alt"]    = str(out_path / f"Altura-VZ.png")

        self.plot_trajectory_3d(show=False, save_path=files["traj3d"])
        self.plot_xy(show=False, save_path=files["xy"], annotate_heading=annotate_heading, stride=stride)
        self.plot_velocities(compare_cmd=True, show=False, save_path=files["vel"], title=None)
        self.plot_altitude(show=False, save_path=files["alt"], overlay_vz=include_vz, smooth_window=smooth_window)

        files["xyz_dualx_steps"] = str(out_path / f"{name}_xyz_dualx_steps.png")  # base
        self.plot_xyz_dual_x_from_step_durations(
            show=False,
            save_path=files["xyz_dualx_steps"],
            separate=False  # genera _x, _y, _z
        )

        return files
    


    def _build_step_time_maps_from_durations(self, N: int):
        """
        Construye mapeos entre Tiempo<->Step usando self.step_durations, sin interpretar nada más.
        Devuelve: (time_to_step, step_to_time, edges, n_steps)

        - edges: tiempos acumulados de inicio de cada step, con edges[0]=0 y edges[-1]=T_total.
        - time_to_step(t): mapea t -> índice de step CONTINUO (k + fracción en [0,1)).
        - step_to_time(s): mapea s -> tiempo continuo (interpolación lineal dentro de cada step).
        """
        import numpy as np

        if not hasattr(self, "step_durations"):
            raise ValueError("[_build_step_time_maps_from_durations] Falta self.step_durations.")

        sd = self.step_durations
        dtc = float(self.dt_control)
        T_end = (N - 1) * dtc  # tiempo del último sample

        # Normaliza a vector de duraciones
        if np.isscalar(sd):
            d = float(sd)
            if d <= 0:
                raise ValueError("step_durations (escala) debe ser > 0.")
            # Número de steps necesarios para cubrir el horizonte de tiempos
            n_steps = max(1, int(np.ceil((N * dtc) / d)))
            durations = np.full(n_steps, d, dtype=float)
        else:
            durations = np.asarray(sd, dtype=float).ravel()
            if durations.size == 0:
                raise ValueError("step_durations (vector) no puede estar vacío.")
            if np.any(durations <= 0):
                raise ValueError("Todos los step_durations deben ser > 0.")
            n_steps = durations.size

        # Bordes acumulados de cada step: [0, d0, d0+d1, ...]
        edges = np.concatenate([[0.0], np.cumsum(durations)])
        # Asegura cubrir al menos hasta T_end
        if edges[-1] < T_end:
            # Si el vector no alcanza, amplía repetendo el último duration para cubrir t (no interpretamos contenido;
            # solo extendemos linealmente para poder mostrar eje; si prefieres, puedes lanzar error aquí).
            extra_needed = T_end - edges[-1]
            n_extra = int(np.ceil(extra_needed / durations[-1]))
            durations = np.concatenate([durations, np.full(n_extra, durations[-1], float)])
            n_steps = durations.size
            edges = np.concatenate([[0.0], np.cumsum(durations)])

        def time_to_step(x):
            x = np.asarray(x, dtype=float)
            idx = np.searchsorted(edges, x, side="right") - 1
            idx = np.clip(idx, 0, n_steps - 1)
            frac = (x - edges[idx]) / durations[idx]
            return idx + frac

        def step_to_time(s):
            s = np.asarray(s, dtype=float)
            k = np.floor(s).astype(int)
            k = np.clip(k, 0, n_steps - 1)
            frac = s - k
            return edges[k] + frac * durations[k]

        return time_to_step, step_to_time, edges, n_steps


    def compute_step_index_from_durations(self):
        """
        Devuelve un vector entero step_idx (longitud N) con el número de step de cada muestra de control,
        usando EXCLUSIVAMENTE self.step_durations.
        """
        import numpy as np
        if not getattr(self, "positions_history", None):
            raise ValueError("[compute_step_index_from_durations] No hay posiciones para obtener N.")
        N = len(self.positions_history)
        dtc = float(self.dt_control)
        t = np.arange(N, dtype=float) * dtc

        time_to_step, _, _, _ = self._build_step_time_maps_from_durations(N)
        step_cont = time_to_step(t)
        # Índice entero por muestra (k de cada tramo)
        step_idx = np.floor(step_cont + 1e-12).astype(int)
        return step_idx


    def plot_xyz_dual_x_from_step_durations(self, show: bool = True, save_path: str | None = None, separate: bool = True):
        """
        Plotea X, Y, Z con doble eje X:
        - Abajo: Tiempo [s] (muestras a dt_control)
        - Arriba: Step [#] (calculado EXACTAMENTE desde self.step_durations)

        separate=True  -> 3 PNG (uno por coordenada) si save_path termina en ".png" (añade _x/_y/_z)
        separate=False -> una sola figura con 3 filas
        """
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator, FuncFormatter

        if not getattr(self, "positions_history", None):
            print("[plot_xyz_dual_x_from_step_durations] No hay datos para graficar.")
            return

        P = np.vstack(self.positions_history)  # (N,3)
        N = P.shape[0]
        dtc = float(self.dt_control)
        t = np.arange(N, dtype=float) * dtc

        # Mapeos basados 100% en self.step_durations
        time_to_step, step_to_time, edges, n_steps = self._build_step_time_maps_from_durations(N)

        # Formateador entero para el eje superior (Step)
        int_formatter = FuncFormatter(lambda v, _: f"{int(round(v))}")

        coord_labels = ["X [m]", "Y [m]", "Z [m]"]
        keys = ["x", "y", "z"]

        if separate:
            files = {}
            for i, k in enumerate(keys):
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(t, P[:, i], linewidth=1.8)
                ax.set_xlabel("Tiempo [s]")
                ax.set_ylabel(coord_labels[i])
                ax.grid(True, alpha=0.6)

                secax = ax.secondary_xaxis('top', functions=(time_to_step, step_to_time))
                secax.set_xlabel("Step [#]")
                secax.xaxis.set_major_locator(MaxNLocator(integer=True))
                secax.xaxis.set_major_formatter(int_formatter)
                # Opcional: limitar a [0, n_steps]
                secax.set_xlim(time_to_step(ax.get_xlim()))

                ax.set_title(f"{coord_labels[i].split()[0]} vs Tiempo (eje superior: Step)")

                if save_path:
                    path_i = save_path.replace(".png", f"_{k}.png")
                    fig.savefig(path_i, dpi=150, bbox_inches="tight")
                    files[k] = path_i
                if show:
                    plt.show()
                plt.close(fig)
            return files if save_path else None

        else:
            fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 9), sharex=True)
            for i, ax in enumerate(axes):
                ax.plot(t, P[:, i], linewidth=1.8)
                if i == 2:
                    ax.set_xlabel("Tiempo [s]")
                ax.set_ylabel(coord_labels[i])
                ax.grid(True, alpha=0.6)

                secax = ax.secondary_xaxis('top', functions=(time_to_step, step_to_time))
                if i == 0:
                    secax.set_xlabel("Step [#]")
                secax.xaxis.set_major_locator(MaxNLocator(integer=True))
                secax.xaxis.set_major_formatter(int_formatter)
                secax.set_xlim(time_to_step(ax.get_xlim()))

            axes[0].set_title("Posición con doble eje X (Tiempo abajo, Step arriba)")
            fig.tight_layout()

            if save_path:
                fig.savefig(save_path, dpi=150, bbox_inches="tight")
            if show:
                plt.show()
            plt.close(fig)




"""
    def plot_xyz_dual_x(self, show: bool = True, save_path: str | None = None, separate: bool = True):

        if not self.positions_history:
            print("[plot_xyz_dual_x] No hay datos para graficar.")
            return

        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator, FuncFormatter

        P = np.vstack(self.positions_history)  # (N, 3)
        
        N = P.shape[0]

        dtc = float(self.dt_control)
        t = np.arange(N, dtype=float) * dtc

        # Relación controles↔step y funciones de mapeo para el eje secundario
        ctrls_per_step, dt_step = self._infer_step_params()

        # Mapeos (continuos) entre tiempo y step (aprox lineal, suficiente para ticks correctos)
        def time_to_step(x):
            return x / dt_step

        def step_to_time(s):
            return s * dt_step

        coord_labels = ["X [m]", "Y [m]", "Z [m]"]
        coord_keys = ["x", "y", "z"]

        if separate:
            files = {}
            for i, key in enumerate(coord_keys):
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(t, P[:, i], linewidth=1.8)
                ax.set_xlabel("Tiempo [s]")
                ax.set_ylabel(coord_labels[i])
                ax.grid(True, alpha=0.6)

                # Eje X secundario arriba con unidades de Step
                secax = ax.secondary_xaxis('top', functions=(time_to_step, step_to_time))
                secax.set_xlabel("Step [#]")
                secax.xaxis.set_major_locator(MaxNLocator(integer=True))
                secax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(round(v))}"))

                ax.set_title(f"{coord_labels[i].split()[0]} vs Tiempo (eje superior: Step)")

                if save_path:
                    path_i = save_path.replace(".png", f"_{key}.png")
                    fig.savefig(path_i, dpi=150, bbox_inches="tight")
                    files[key] = path_i
                if show:
                    plt.show()
                plt.close(fig)
            return files if save_path else None

        else:
            fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 9), sharex=True)
            for i, ax in enumerate(axes):
                ax.plot(t, P[:, i], linewidth=1.8)
                if i == 2:
                    ax.set_xlabel("Tiempo [s]")
                ax.set_ylabel(coord_labels[i])
                ax.grid(True, alpha=0.6)

                secax = ax.secondary_xaxis('top', functions=(time_to_step, step_to_time))
                if i == 0:
                    secax.set_xlabel("Step [#]")
                secax.xaxis.set_major_locator(MaxNLocator(integer=True))
                secax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(round(v))}"))

            axes[0].set_title("Posición con doble eje X (Tiempo abajo, Step arriba)")
            fig.tight_layout()

            if save_path:
                fig.savefig(save_path, dpi=150, bbox_inches="tight")
            if show:
                plt.show()
            plt.close(fig)

"""