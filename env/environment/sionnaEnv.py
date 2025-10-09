from __future__ import annotations
import math
import numpy as np
from pathlib import Path
import math

from sionna.rt import (
    Scene,
    Transmitter,
    Receiver,
    PlanarArray,
    load_scene,
    PathSolver,
    scene as builtin_scenes,
    Camera,
    RadioMapSolver,
)

from sionna.sys import PHYAbstraction, OuterLoopLinkAdaptation, PFSchedulerSUMIMO, downlink_fair_power_control
from sionna.phy.nr.utils import decode_mcs_index
from sionna.phy.utils import log2, dbm_to_watt, lin_to_db
from sionna.phy.constants import BOLTZMANN_CONSTANT

import tensorflow as tf
import logging
tf.get_logger().setLevel(logging.ERROR)
try:
    import absl.logging
    absl.logging.set_verbosity('error')
except Exception:
    pass

import numpy as np

from sionna.phy.ofdm import ResourceGrid, RZFPrecodedChannel, LMMSEPostEqualizationSINR
from sionna.phy.mimo import StreamManagement
from sionna.sys.utils import spread_across_subcarriers
from sionna.rt import subcarrier_frequencies




# ---------- Utilidades ----------
def _resolve_scene_path(scene_id: str) -> str | None:
    """Devuelve ruta a XML o GLB si 'scene_id' es válido en scenes/ o Mapas-pruebas/."""
    from pathlib import Path

    p = Path(scene_id)
    if p.exists():
        return str(p)

    # Buscar en scenes/
    base = Path(__file__).resolve().parents[1] / "scenes"
    cand_xml = base / f"{scene_id}.xml"
    if cand_xml.exists():
        return str(cand_xml)
    cand_dir = base / scene_id
    if cand_dir.is_dir():
        xml = cand_dir / "scene.xml"
        if xml.exists():
            return str(xml)

    # Buscar en Mapas-pruebas/
    base_maps = Path(__file__).resolve().parents[2] / "Mapas-pruebas"
    # Si ya viene con extensión (ej: plaza.glb), lo busca directo
    cand_file = base_maps / scene_id
    if cand_file.exists():
        return str(cand_file)
    # Si viene sin extensión, probar .glb
    cand_glb = base_maps / f"{scene_id}.glb"
    if cand_glb.exists():
        return str(cand_glb)

    return None


def load_builtin_scene(name: str = "munich",
                       frequency_hz: float = 3.5e9,
                       merge_shapes: bool = True):
    """Carga una escena integrada y devuelve (scene, solver)."""
    scene_obj = None
    if hasattr(builtin_scenes, name):
        scene_obj = getattr(builtin_scenes, name)
    scene = load_scene(scene_obj if scene_obj is not None else name,
                       merge_shapes=merge_shapes)
    scene.frequency = frequency_hz

    # Valores seguros por defecto; se pueden sobreescribir en build_scene()
    scene.tx_array = PlanarArray(
        num_rows=1, num_cols=1,
        vertical_spacing=0.5, horizontal_spacing=0.5,
        pattern="iso", polarization="V"
    )
    scene.rx_array = PlanarArray(
        num_rows=1, num_cols=1,
        vertical_spacing=0.5, horizontal_spacing=0.5,
        pattern="iso", polarization="V"
    )

    solver = PathSolver()
    return scene, solver





# ---------- Wrapper Sionna RT ----------

"""
Defaults del PathSolver: {
    'samples_per_src': 1000000, 'max_num_paths_per_src': 1000000, 'synthetic_array': True, 'max_depth': 3,
    'los': True, 'specular_reflection': True, 'diffuse_reflection': False, 'refraction': True, 'seed': 42}
"""
class SionnaRT:
    """Gestión de escena y métricas con Sionna RT (sin fallback)."""

    def __init__(self,
                 # --- Modo de antena del dron ---
                 antenna_mode: str = "ISO",       # "ISO" | "SECTOR3_3GPP"

                 # --- RF / ruido ---
                 frequency_mhz: float = 3500.0,   # Frecuencia portadora [MHz]
                 tx_power_dbm: float = 40.0,      # Potencia TOTAL objetivo [dBm] (se reparte si hay sectores)
                 
                 bandwidth_hz: float = 18_000_000,      # Ancho de banda de ruido térmico [Hz]

                 # --- escena: nombre integrado o ruta a XML/carpeta ---
                 scene_name: str = "munich",

                 # --- antenas TX (matriz global de la escena) ---
                 tx_array_rows: int = 1,          # Nº de filas de la matriz TX
                 tx_array_cols: int = 1,          # Nº de columnas de la matriz TX
                 tx_array_v_spacing: float = 0.5, # Separación vertical (en λ)
                 tx_array_h_spacing: float = 0.5, # Separación horizontal (en λ)
                 tx_array_pattern: str = "iso",   # "iso","dipole","tr38901", etc.
                 tx_array_polarization: str = "V",# "V","H","VH" (dual)

                 # --- antenas RX (matriz global de la escena) ---
                 rx_array_rows: int = 1,
                 rx_array_cols: int = 1,
                 rx_array_v_spacing: float = 0.5,
                 rx_array_h_spacing: float = 0.5,
                 rx_array_pattern: str = "iso",
                 rx_array_polarization: str = "V",

                 # --- pose inicial del transmisor ---
                 tx_initial_position: tuple[float, float, float] = (0.0, 0.0, 10.0), # [m]
                 tx_orientation_deg: tuple[float, float, float] = (0.0, -90.0, 0.0), # [°] yaw,pitch,roll

                 # --- control del trazador de caminos (PathSolver) ---
                 max_depth: int = 5,              # Nº máx. de interacciones por camino
                 los: bool = True,                # Considerar Line-of-Sight
                 specular_reflection: bool = True,# Reflexiones especulares (reflexiones tipo espejo)
                 diffuse_reflection: bool = True, # Reflexiones difusas, por superficies rugosas (muy costoso, realista)
                 refraction: bool = True,         # Refracción (atravesar vidrios, etc. cambiar angulo y atenuar)
                 synthetic_array: bool = False,   # True: matriz sintética (rápido); False: por elemento (en false realista)
                 samples_per_src: int | None = 500_000,    # Nº de rayos por fuente (default 1,000,000)
                 max_num_paths_per_src: int | None = None,  # Tope de caminos por fuente (None => default) (default 1000000)
                 seed: int = 41,                   # Semilla del muestreo estocástico



                # --- parametros SYS
                num_ut: int = 6,                # número de usuarios/receptores
                num_subcarriers: int = 128,     # número de subportadoras
                num_ofdm_symbols: int = 12,     # número de símbolos OFDM
                bler_target: float = 0.1,       # objetivo de BLER para el enlace
                mcs_table_index: int = 1,       # índice de la tabla MCS a utilizar
                num_ut_ant: int = 1,            # número de antenas por usuario
                num_bs: int = 1,                # número de estaciones base                
                subcarrier_spacing: float = 30e3,
               ):

        # --- Modo ---
        self.antenna_mode = str(antenna_mode).upper()

        # --- RF / ruido ---
        self.freq_hz = frequency_mhz * 1e6
        self.tx_power_dbm_total = tx_power_dbm     
        self.bandwidth_hz = bandwidth_hz

        # --- escena / antenas ---
        self.scene_name = scene_name


        # --- configuracion antenas TX ---
        self.tx_array_rows = tx_array_rows
        self.tx_array_cols = tx_array_cols
        self.tx_array_v_spacing = tx_array_v_spacing
        self.tx_array_h_spacing = tx_array_h_spacing
        self.tx_array_pattern = tx_array_pattern
        self.tx_array_polarization = tx_array_polarization

        # --- configuracion receptores RX ---
        self.rx_array_rows = rx_array_rows
        self.rx_array_cols = rx_array_cols
        self.rx_array_v_spacing = rx_array_v_spacing
        self.rx_array_h_spacing = rx_array_h_spacing
        self.rx_array_pattern = rx_array_pattern
        self.rx_array_polarization = rx_array_polarization

        # --- pose inicial del transmisor ---
        self.tx_initial_position = tx_initial_position
        self.tx_orientation_deg = tx_orientation_deg  # yaw, pitch, roll en grados

        # --- PathSolver ---
        self.max_depth = max_depth
        self.los = los
        self.specular_reflection = specular_reflection
        self.diffuse_reflection = diffuse_reflection
        self.refraction = refraction
        self.synthetic_array = synthetic_array
        self.samples_per_src = samples_per_src
        self.max_num_paths_per_src = max_num_paths_per_src
        self.seed = seed

        # --- objetos RT ---
        self.scene: Scene | None = None
        self._solver: PathSolver | None = None
        self.tx: Transmitter | None = None  # primer TX (para compatibilidad)
        self.txs: list[Transmitter] = []    # lista completa de TX
        self.rx_list: list[Receiver] = []



        # Guarda parámetros que usaremos para SYS
        self.num_ut = num_ut
        self.num_subcarriers = num_subcarriers
        self.num_ofdm_symbols = num_ofdm_symbols
        self.bler_target = bler_target
        self.mcs_table_index = mcs_table_index
        self.num_ut_ant = num_ut_ant
        self.num_bs = num_bs
        self.subcarrier_spacing = subcarrier_spacing



        # PHY Abstraction
        self.phy_abs = PHYAbstraction()

        # OLLA Link Adaptation (Outer Loop)
        self.olla = OuterLoopLinkAdaptation(
            self.phy_abs,
            num_ut=self.num_ut,
            bler_target=self.bler_target,
            batch_size=[self.num_bs]
        )

        # Scheduler: Proporcional Fair Scheduling para Sumimo
        self.scheduler = PFSchedulerSUMIMO(
            num_ut=self.num_ut,
            num_freq_res=self.num_subcarriers,
            num_ofdm_sym=self.num_ofdm_symbols,
            batch_size=[self.num_bs],
            num_streams_per_ut=self.num_ut_ant,
            beta=0.9            # factor para eleccion de receptor 0 lo mas distribuido posible, 1 el que menos le a tocado
        )

        
        # Crear ResourceGrid para usar después
        self.resource_grid = ResourceGrid(
            num_ofdm_symbols=self.num_ofdm_symbols,
            fft_size=self.num_subcarriers,
            subcarrier_spacing=self.subcarrier_spacing,
            num_tx=self.num_ut,               # número de transmisores
            num_streams_per_tx=self.num_ut_ant
        )

        self.stream_management = StreamManagement(
            tf.ones([self.num_ut, self.num_bs]), self.num_ut * self.num_ut_ant
        )

        self.precoded_channel = RZFPrecodedChannel(
            resource_grid=self.resource_grid,
            stream_management=self.stream_management
        )

        self.lmmse_posteq_sinr = LMMSEPostEqualizationSINR(
            resource_grid=self.resource_grid,
            stream_management=self.stream_management
        )

        self.harq_feedback = -tf.ones([self.num_bs, self.num_ut], dtype=tf.int32)
        self.sinr_eff_feedback = tf.zeros([self.num_bs, self.num_ut], dtype=tf.float32)
        self.num_decoded_bits = tf.zeros([self.num_bs, self.num_ut], dtype=tf.int32)

    # ---- Construcción ----
    def build_scene(self):
        xml_path = _resolve_scene_path(self.scene_name)

        if xml_path is not None:
            if xml_path.endswith((".glb", ".gltf", ".obj")):
                # Escena externa (ej: Santiago)
                scene = load_scene(xml_path, merge_shapes=True)
            else:
                # Escena XML estándar
                scene = load_scene(xml_path, merge_shapes=True)
        else:
            # Escena interna de Sionna
            scene, _ = load_builtin_scene(name=self.scene_name,
                                        frequency_hz=self.freq_hz,
                                        merge_shapes=True)
            

        self.scene = scene
        self.scene.frequency = self.freq_hz
        pmin, pmax = self.scene_bounds_xyz()

        # 🔹 Guardamos los límites de la escena para que DroneEnv pueda usarlos
        self.scene_bounds = (pmin, pmax)

        # Configura arrays globales (se aplican a todos los TX/RX)
        # Si el modo es SECTOR3_3GPP y el usuario dejó "iso", forzamos patrón 3GPP:
        tx_pattern = self.tx_array_pattern
        tx_pol = self.tx_array_polarization
        if self.antenna_mode in ("SECTOR3_3GPP", "SECTOR3", "3GPP"):
            if tx_pattern == "iso":
                tx_pattern = "tr38901"
            if tx_pol == "V":
                tx_pol = "VH"  # dual, más cercano a 5G

        self.scene.tx_array = PlanarArray(
            num_rows=self.tx_array_rows, num_cols=self.tx_array_cols,
            vertical_spacing=self.tx_array_v_spacing,
            horizontal_spacing=self.tx_array_h_spacing,
            pattern=tx_pattern, polarization=tx_pol
        )
        self.scene.rx_array = PlanarArray(
            num_rows=self.rx_array_rows, num_cols=self.rx_array_cols,
            vertical_spacing=self.rx_array_v_spacing,
            horizontal_spacing=self.rx_array_h_spacing,
            pattern=self.rx_array_pattern, polarization=self.rx_array_polarization
        )

        # Precoder/combiner off si existen
        if hasattr(self.scene, "transmit_precoder"):
            self.scene.transmit_precoder = None
        if hasattr(self.scene, "receive_combiner"):
            self.scene.receive_combiner = None

        self._solver = PathSolver()

        # Transmisores según modo
        self._create_transmitters()

        # Sanity
        assert self.scene is not None and self._solver is not None and self.tx is not None, \
            "Sionna RT no quedó inicializado correctamente."



    def _create_transmitters(self):
        """
        Crea SIEMPRE 1 TX y lo añade a la escena.
        - Modo ISO: patrón isotrópico (ya seteado en scene.tx_array), orientación base.
        - Modo 3GPP/SECTOR3/SECTOR3_3GPP: patrón 3GPP TR 38.901 (ya seteado en scene.tx_array),
        usa un downtilt (pitch negativo) para 'apuntar hacia abajo'.
        Efectos colaterales:
            self.txs = [tx]
            self.tx  = tx
        """
        # Limpia lista local (si re-llamas esta función no duplicas referencias)
        self.txs = []

        # Helpers
        def _norm_deg(a: float) -> float:
            x = float(a) % 360.0
            return x if x >= 0.0 else x + 360.0

        # Orientación base [yaw, pitch, roll]
        try:
            base_yaw, base_pitch, base_roll = self.tx_orientation_deg
        except Exception as e:
            raise ValueError("tx_orientation_deg debe ser [yaw, pitch, roll] en grados") from e

        # Si no definiste un downtilt, usamos -10° (negativo = inclinar hacia el suelo)
        # ajusta a tu convención si tu motor usa el signo inverso.
        tx_downtilt_deg = getattr(self, "tx_downtilt_deg", -10.0)

        # Crea el único TX
        tx = Transmitter(
            name="tx0",
            position=list(self.tx_initial_position),
            display_radius=2
        )

        mode = str(self.antenna_mode).upper().strip()

        if mode == "ISO":
            # Un solo TX, toda la potencia, orientación base
            tx.orientation = [_norm_deg(base_yaw), float(base_pitch), float(base_roll)]
            tx.power_dbm = float(self.tx_power_dbm_total)

        elif mode in ("SECTOR3_3GPP", "SECTOR3", "3GPP"):
            # Un solo TX, patrón 3GPP ya viene dado por scene.tx_array en build_scene.
            # Usamos downtilt (pitch negativo) para 'apuntar hacia abajo'.
            # Nota: si tu convención es opuesta, cambia a +abs(tx_downtilt_deg).
            pitch = float(base_pitch) + float(tx_downtilt_deg)  # típico: base 0 + (-10) = -10°
            tx.orientation = [_norm_deg(base_yaw), pitch, float(base_roll)]
            tx.power_dbm = float(self.tx_power_dbm_total)  # sin reparto, 1 solo sector físico

            # (Opcional) si quieres permitir un azimut distinto para el lóbulo principal:
            # desired_azimuth = getattr(self, "tx_azimuth_deg", base_yaw)
            # tx.orientation[0] = _norm_deg(desired_azimuth)

        else:
            raise ValueError(f"antenna_mode inválido: {self.antenna_mode}")

        # Añade a escena y guarda referencias
        self.scene.add(tx)
        self.txs.append(tx)
        self.tx = tx  # compat




    def attach_receivers(self, rx_positions_xyz: np.ndarray):
        assert self.scene is not None, "build_scene() no fue llamado."
        self.rx_list = []
        for i, p in enumerate(rx_positions_xyz):
            rx = Receiver(name=f"RX_{i}",
                          position=[float(p[0]), float(p[1]), float(p[2])],
                          display_radius=1.5, color=(0, 0, 0))
            self.scene.add(rx)
            self.rx_list.append(rx)

    def move_tx(self, pos_xyz):
        """Mueve TODOS los TX (1 o 3) a la misma posición del dron."""
        assert self.txs, "TX no inicializados."
        pos = [float(pos_xyz[0]), float(pos_xyz[1]), float(pos_xyz[2])]
        for tx in self.txs:
            tx.position = pos
        # Mantiene orientaciones según el modo (no se recalculan aquí)



    def scene_bounds_xyz(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Devuelve (min_xyz, max_xyz) de la escena en metros.
        Requiere que self.scene ya esté construido por build_scene().
        """
        # Mitsuba Scene subyacente
        mi_scene = self.scene.mi_scene            # Sionna -> Mitsuba scene

        # Bounding box global de la escena
        bb = mi_scene.bbox()                      # mi.BoundingBox3f  (min, max)

        pmin = np.array([float(bb.min.x), float(bb.min.y), float(bb.min.z)], dtype=float)
        pmax = np.array([float(bb.max.x), float(bb.max.y), float(bb.max.z)], dtype=float)
        return pmin, pmax

    # ---- Cálculo de paths y métricas ----
    def _paths(self):
        assert self.scene is not None and self._solver is not None, "Escena/solver no inicializados."
        extra = {}
        if self.samples_per_src is not None:
            extra["samples_per_src"] = int(self.samples_per_src)
        if self.max_num_paths_per_src is not None:
            extra["max_num_paths_per_src"] = int(self.max_num_paths_per_src)

        return self._solver(
            scene=self.scene,
            max_depth=self.max_depth,
            los=self.los,
            specular_reflection=self.specular_reflection,
            diffuse_reflection=self.diffuse_reflection,
            refraction=self.refraction,
            synthetic_array=self.synthetic_array,
            seed=self.seed,
            **extra,
        )

    def _total_tx_power_dbm(self) -> float:
        """Suma la potencia de TODOS los TX activos (en dBm), casteando tipos drjit a float."""
        # Si no hay lista de TX creada aún, usa la potencia total configurada
        if not self.txs:
            return float(self.tx_power_dbm_total)

        p_mw = 0.0
        for tx in self.txs:
            try:
                # power_dbm puede ser drjit.Float; lo convertimos a float de Python
                p_dbm = float(tx.power_dbm)
            except Exception:
                p_dbm = float(self.tx_power_dbm_total)
            p_mw += 10.0 ** (p_dbm / 10.0)

        # Evita log10(0) y asegura tipo nativo
        p_mw = float(p_mw)
        return 10.0 * math.log10(p_mw if p_mw > 0.0 else 1e-30)


    def compute_prx_dbm(self) -> np.ndarray:
        """
        PRx[dBm] por receptor usando CIR de Sionna RT.
        Suma |a|^2 a través de TODAS las dimensiones excepto RX y escala por
        la potencia TOTAL de los TX activos (1 u 3).
        """
        paths = self._paths()
        a, _ = paths.cir(out_type="numpy")  # a: compleja
        a = np.atleast_1d(a)
        if a.ndim == 0:
            power_lin = np.array([np.abs(a) ** 2], dtype=float)
        else:
            axes_to_sum = tuple(i for i in range(a.ndim) if i != 0)
            power_lin = np.sum(np.abs(a) ** 2, axis=axes_to_sum)

        power_lin = np.maximum(power_lin.astype(float), 1e-24)
        ptx_dbm_total = self._total_tx_power_dbm()
        prx_dbm = ptx_dbm_total + 10.0 * np.log10(power_lin)
        return prx_dbm.astype(float)



    # ---- Visualización opcional ----
    def preview_scene(self):
        assert self.scene is not None, "Scene no inicializada."
        try:
            self.scene.preview()
        except Exception as e:
            print("preview() requiere Jupyter. Usa render_scene_to_file().", e)

    def render_scene_to_file(self, filename: str = "scene.png",
                             resolution: tuple[int, int] = (900, 700),
                             with_radio_map: bool = False) -> bool:
        assert self.scene is not None, "Scene no inicializada."

        cam = self._auto_camera()
        try:
            if with_radio_map:
                rm_solver = RadioMapSolver()
                rm = rm_solver(scene=self.scene, max_depth=self.max_depth,
                               cell_size=[1, 1], samples_per_tx=10**5)
                self.scene.render_to_file(camera=cam, radio_map=rm,
                                          filename=filename, resolution=list(resolution))
            else:
                self.scene.render_to_file(camera=cam, filename=filename,
                                          resolution=list(resolution))
            print(f"Imagen guardada en: {filename}")
            return True
        except Exception as e:
            print("Error al renderizar la escena:", e)
            return False

    def _auto_camera(self, z_scale: float = 1.6) -> Camera:

        try:
            aabb = self.scene_bounds
            mn, mx = aabb[0], aabb[1]
            cx = float((mn[0] + mx[0]) / 2)
            cy = float((mn[1] + mx[1]) / 2)
            size_xy = max(float(mx[0] - mn[0]), float(mx[1] - mn[1]))
            z = max(150.0, size_xy * z_scale)
            return Camera(position=[cx, cy, z], look_at=[cx, cy, 0.0])
        except Exception:
            # Fallback fijo
            print("No se pudo calcular cámara automáticamente, usando fallback.")
            return Camera(position=[0, 0, 300], look_at=[0, 0, 0])
        


    # --- Calculo de SYS
    @tf.function  # (jit_compile=True)
    def sys_step(self, h, harq_feedback, sinr_eff_feedback, num_decoded_bits):
        """
        Ejecuta un step del sistema usando Sionna SYS.
        Calcula y devuelve métricas de bloques por step y acumuladas.
        """
        # --- Ruido (potencia de ruido según BW efectiva) ---
        # --- Ruido por subportadora en dBm (FIJO razonable) ---
        no_dbm_sc = tf.constant(-129.0, tf.float32)            # 30 kHz típico
        no = tf.pow(10.0, no_dbm_sc/10.0) / 1e3                # -> Watts por subportadora
        EPS_NO = tf.constant(1e-18, tf.float32)
        no = tf.maximum(no, EPS_NO)



        # --- Ganancia de canal y tasa Shannon estimada (para scheduler PF) ---
        # h: [num_ut, num_ut_ant, num_bs, num_bs_ant, T, F]
        channel_gain = tf.maximum(tf.math.square(tf.abs(h)), 1e-12)   

        # log2(1 + SNR) por RE/antena
        rate = log2(1.0 + channel_gain / no)  # [ut, ut_ant, bs, bs_ant, T, F]

        # Promedio sobre antenas de UT y BS (¡solo antenas!)
        rate = tf.reduce_mean(rate, axis=[1, 3])  # -> [num_ut, num_bs, T, F]

        # Reordenar a lo que espera el scheduler
        rate_achievable_est = tf.transpose(rate, [1, 2, 3, 0])  # -> [num_bs, T, F, num_ut]


        # --- Scheduler (PF) ---
        # is_scheduled: [num_bs, num_ofdm_symbols, num_subcarriers, num_ut, num_streams_per_ut]
        is_scheduled = self.scheduler(num_decoded_bits, rate_achievable_est)

        # REs asignados por BS y UT: [num_bs, num_ut]
        num_allocated_re = tf.reduce_sum(tf.cast(is_scheduled, tf.int32), axis=[ 1, 2, 4])

        # --- Pathloss medio por UT (heurístico) ---
        pathloss_per_ut = tf.reduce_mean(1.0 / channel_gain, axis=[1, 3, 4, 5])  # [num_ut, num_bs]
        pathloss_per_ut = tf.transpose(pathloss_per_ut, [1, 0])                  # [num_bs, num_ut]

        pathloss_per_ut = tf.where(tf.math.is_finite(pathloss_per_ut),
                           pathloss_per_ut,
                           tf.reduce_max(pathloss_per_ut[tf.math.is_finite(pathloss_per_ut)]) )


        # --- Control de potencia DL ---
        tx_power_per_ut, _ = downlink_fair_power_control(
            pathloss_per_ut, no, num_allocated_re,
            bs_max_power_dbm=self.tx_power_dbm_total,
            guaranteed_power_ratio=0.5,
            fairness=0
        )
        tx_power_per_ut = tf.nn.relu(tx_power_per_ut)  # no negativos
        

        # --- Reparto de potencia en REs asignados ---
        tx_power = spread_across_subcarriers(
            tf.expand_dims(tx_power_per_ut, axis=-2),  # [num_bs, num_ut, 1]
            is_scheduled,
            num_tx=self.num_bs
        )

       
        # Precoding con regularización más alta (evita mal acondicionamiento)
        h_eff = self.precoded_channel(h[tf.newaxis, ...],
                                    tx_power=tx_power,
                                    alpha=no*20.0 + EPS_NO)

        # SINR sin whitening (evita Cholesky)
        sinr = self.lmmse_posteq_sinr(h_eff, no=no + EPS_NO, interference_whitening=False)


        # sinr: [num_bs, num_ofdm_symbols, num_subcarriers, num_ut, num_streams_per_ut]

        # --- Link Adaptation (OLLA) ---
        mcs_index = self.olla(
            num_allocated_re=num_allocated_re,
            sinr_eff=sinr_eff_feedback,
            mcs_table_index=self.mcs_table_index,
            mcs_category=1,  # downlink
            harq_feedback=harq_feedback
        )

        # --- Abstracción PHY: bits decodados, HARQ, SINR efectivo real ---
        num_decoded_bits, harq_feedback, sinr_eff_true, *_ = self.phy_abs(
            mcs_index, sinr=sinr, mcs_table_index=self.mcs_table_index, mcs_category=1
        )

        # === BLOQUE NUEVO: HARQ masked al estilo tutorial (unscheduled=-1) ===
        # num_allocated_re: [num_bs, num_ut]
        harq_feedback_masked = tf.where(
            num_allocated_re > 0,                 # programado en este step
            harq_feedback,                        # 1=ACK / 0=NACK
            -tf.ones_like(harq_feedback)          # -1 si NO fue agendado (tutorial)
        )  # -> [num_bs, num_ut]


        sinr_eff_db_true = lin_to_db(sinr_eff_true)  # Effective SINR
        # feedback para OLLA en el próximo step (0 si no fue agendado)
        sinr_eff_feedback = tf.where(num_allocated_re > 0, sinr_eff_true, 0)

        # --- Eficiencias ---
        mod_order, coderate = decode_mcs_index(
            mcs_index, table_index=self.mcs_table_index, is_pusch=False
        )

        # SE simulada por link adaptation (solo si ACK)
        se_la = tf.where(
            harq_feedback == 1,
            tf.cast(mod_order, coderate.dtype) * coderate,
            tf.cast(0, tf.float32)
        )
        se_shannon = log2(1.0 + sinr_eff_true)  # Cota superior de SE


        # --- TBLER por usuario (Transport Block) ------------------------------
      
      
        # Indicador de TB “real” según HARQ (tutorial-style)
        tb_tx_step_per_ue = tf.cast(tf.not_equal(harq_feedback[0, :], -1), tf.int32)  # 1 si hubo intento de TB

        # Éxito del TB (ACK) entre los que sí transmitieron
        tb_ok_step_per_ue = tf.cast(tf.equal(harq_feedback[0, :], 1), tf.int32) * tb_tx_step_per_ue

        # TBLER step: 0 (ACK), 1 (NACK), NaN (no transmitió)
        tbler_step_per_ue = tf.where(
            tb_tx_step_per_ue > 0,
            1.0 - tf.cast(tb_ok_step_per_ue, tf.float32),
            tf.constant(float('nan'), dtype=tf.float32)
        )



        # Media del step SOLO sobre UEs que transmitieron (ignora NaN)
        tbler_step_per_ue = tbler_step_per_ue[tf.newaxis, :]  # shape [1, num_ut] para coherencia de salida


        # --- Contadores de bloques para acumulados ----------------------------
        # Permiten TBLER acumulada = 1 - blocks_ok_accum / blocks_tx_accum
        blocks_tx_step_per_ue = tb_tx_step_per_ue
        blocks_ok_step_per_ue = tb_ok_step_per_ue

        step_blocks = {
            "blocks_tx_step_per_ue": blocks_tx_step_per_ue,
            "blocks_ok_step_per_ue": blocks_ok_step_per_ue,
            
        }

       
        return (
            harq_feedback, sinr_eff_feedback, num_decoded_bits,
            se_la, se_shannon, sinr_eff_db_true, 
            tbler_step_per_ue,  
            step_blocks, harq_feedback_masked
        )




    def get_current_channel_tensor(self) -> tf.Tensor:
        if not hasattr(self, 'current_cfr_for_current_step'):
            raise ValueError("No se ha actualizado current_cfr_for_current_step aún.")
        return self.current_cfr_for_current_step



    def run_sys_step(self):
        # Actualiza CFR para el step actual y obtén tensor de canal
        self.update_and_store_cfr_for_step()
        h = self.get_current_channel_tensor()


        (self.harq_feedback,
        self.sinr_eff_feedback,
        self.num_decoded_bits,
        se_la,
        se_shannon,
        sinr_eff_db_true,
        tbler_per_user,            # << renombrado (antes bler_per_user)
        step_blocks,
        harq_feedback_masked) = self.sys_step(
            h=h,
            harq_feedback=self.harq_feedback,
            sinr_eff_feedback=self.sinr_eff_feedback,
            num_decoded_bits=self.num_decoded_bits
        )

        # ---------- Inicializar acumuladores si no existen ----------
        if not hasattr(self, "blocks_acc_tx"):
            self.blocks_acc_tx = [0 for _ in range(self.num_ut)]
        if not hasattr(self, "blocks_acc_ok"):
            self.blocks_acc_ok = [0 for _ in range(self.num_ut)]
        if not hasattr(self, "bits_acc_total"):
            self.bits_acc_total = [0 for _ in range(self.num_ut)]

        # ---------- Pasar a numpy para armar métricas ----------
        sinr_eff_db_true_np = sinr_eff_db_true.numpy()[0, :]  # [num_ut]
        se_la_np = se_la.numpy()[0, :]                        # [num_ut]
        se_shannon_np = se_shannon.numpy()[0, :]              # [num_ut]        
        tbler_np = tbler_per_user.numpy()[0, :]               # [num_ut]  (0/1/NaN)
        bits_np = self.num_decoded_bits.numpy()[0, :]         # [num_ut]

        blocks_tx_step_per_ue_np = step_blocks["blocks_tx_step_per_ue"].numpy()          # [num_ut]
        blocks_ok_step_per_ue_np = step_blocks["blocks_ok_step_per_ue"].numpy()          # [num_ut]

        # ---------- Acumular ----------
        for i in range(self.num_ut):
            self.blocks_acc_tx[i] += int(blocks_tx_step_per_ue_np[i])
            self.blocks_acc_ok[i] += int(blocks_ok_step_per_ue_np[i])
            self.bits_acc_total[i] += int(bits_np[i])

       

        # ---------- Métricas por-UE ----------
        ue_metrics = []
        prx_list = self.compute_prx_dbm()
        for i in range(self.num_ut):
            se_la_i       = float(se_la_np[i])
            se_shannon_i  = float(se_shannon_np[i])
            if se_shannon_i > 0.0:
                se_gap_pct_i = max(0.0, (1.0 - (se_la_i / se_shannon_i)) * 100.0)
            else:
                se_gap_pct_i = float('nan')

            ue_metrics.append({
                "ue_id": i, #
                "sinr_eff_db": float(sinr_eff_db_true_np[i]), #
                "prx_dbm": float(prx_list[i]), 
                "se_la": float(se_la_np[i]), #
                "se_shannon": float(se_shannon_np[i]), #
                "se_gap_pct": se_gap_pct_i, #

                # Métrica TBLER del step por UE (0/1/NaN si no transmitió)
                "tbler": float(tbler_np[i]), #
            })

        


        # ---------- Historial HARQ estilo tutorial ----------
        if not hasattr(self, "harq_feedback_hist"):
            self.harq_feedback_hist = []  # lista de arrays [num_bs, num_ut] con -1/0/1

        harq_mask_np = harq_feedback_masked.numpy()      # [num_bs, num_ut]
        self.harq_feedback_hist.append(harq_mask_np)     # apéndice por step

        # ---------- TBLER running por UE (idéntico a tutorial) ----------
        # Construye tensor [S, num_bs, num_ut]
        harq_hist_np = np.stack(self.harq_feedback_hist, axis=0)

        # Usamos BS=0 para la curva clásica (puedes extender a multi-BS si lo necesitas)
        harq_bs0 = harq_hist_np[:, 0, :]                 # [S, num_ut]

        # Reemplaza -1 (unscheduled) por NaN para usar nancumsum/nanmean
        harq_bs0_nan = harq_bs0.astype(float)
        harq_bs0_nan[harq_bs0_nan == -1] = np.nan        # ACK=1, NACK=0, UNSCHED=NaN

        # Acumulados de TX y OK
        tx_cum = np.cumsum(~np.isnan(harq_bs0_nan), axis=0).astype(float)  # [S, num_ut]
        ok_cum = np.nancumsum(harq_bs0_nan, axis=0)                        # [S, num_ut]

        # TBLER running = 1 - OK_acum / TX_acum
        tbler_running_per_ue = 1.0 - (ok_cum / np.maximum(tx_cum, 1.0))    # [S, num_ut]
        tbler_running_per_ue_step = tbler_running_per_ue[-1, :]            # [num_ut]


        return {
            "ue_metrics": ue_metrics,
            

            # === NUEVO: iguales al tutorial, por UE (y un promedio global útil) ===
            "tbler_running_per_ue": tbler_running_per_ue_step.tolist(),            
        }



    def update_and_store_cfr_for_step(self):
        """
        Calcula el CFR para el step actual y lo deja listo para SYS
        con forma [num_ut, num_ut_ant, num_bs, num_bs_ant, T, F].
        """
        # 1) Paths y frecuencias
        paths = self._paths()
        frequencies = subcarrier_frequencies(
            num_subcarriers=self.num_subcarriers,
            subcarrier_spacing=self.subcarrier_spacing
        )

        # 2) CFR crudo desde RT (numpy): [num_rx, num_tx, T, F]
        h_freq_np = paths.cfr(
            frequencies=frequencies,
            sampling_frequency=1 / self.resource_grid.ofdm_symbol_duration,
            num_time_steps=self.num_ofdm_symbols,
            out_type="numpy"
        )

        # 3) Formateo robusto a layout SYS y guardado
        h_tf = tf.convert_to_tensor(h_freq_np, dtype=tf.complex64)
        
        self.current_cfr_for_current_step = h_tf



    