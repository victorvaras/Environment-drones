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

from sionna.sys import PHYAbstraction, OuterLoopLinkAdaptation, downlink_fair_power_control
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


def tf_unwrap_phase(phi: tf.Tensor) -> tf.Tensor:
    two_pi = tf.constant(2.0*np.pi, dtype=phi.dtype)
    d = phi[1:] - phi[:-1]
    d_adj = d - two_pi * tf.round(d / two_pi)
    phi0 = phi[0:1]
    phi_tail = phi0 + tf.cumsum(d_adj)
    return tf.concat([phi0, phi_tail], axis=0)

def _slope_and_fd_for_one_f(x_t: tf.Tensor, Tsym: tf.Tensor):
    phi = tf.math.angle(x_t)
    phi_unw = tf_unwrap_phase(phi)
    T = tf.shape(phi_unw)[0]
    slope = (phi_unw[-1] - phi_unw[0]) / tf.cast(tf.maximum(1, T-1), tf.float32)  # rad/símb
    two_pi = tf.constant(2.0*np.pi, tf.float32)
    fD = slope / (two_pi * Tsym)  # Hz
    return slope, fD

def _median_1d(x: tf.Tensor):
    x_sorted = tf.sort(x, axis=0)
    n = tf.shape(x_sorted)[0]
    mid = n // 2
    return tf.cond(
        tf.equal(n % 2, 1),
        lambda: x_sorted[mid],
        lambda: 0.5*(x_sorted[mid-1] + x_sorted[mid])
    )

def doppler_metrics_multi(
    h: tf.Tensor,                       # [UT, Uant, BS, Bant, T, F] complex64
    ofdm_symbol_duration_s: float,      # Tsym (s)
    scs_hz: float,                      # SCS (Hz)
    f_indices: tuple = (5, 20, 40, 60, 80, 100),
    avg_over_antennas: bool = True,     # promediar sobre Uant/Bant
    use_median_over_f: bool = True,     # mediana sobre subportadoras elegidas
):
    Tsym = tf.convert_to_tensor(ofdm_symbol_duration_s, tf.float32)
    scs  = tf.convert_to_tensor(scs_hz, tf.float32)

    # Promedio sobre antenas si quieres robustez: [UT, BS, T, F]
    if avg_over_antennas:
        H = tf.reduce_mean(h, axis=(1,3))  # [UT, BS, T, F]
        has_ant = False
    else:
        H = h                               # [UT, Uant, BS, Bant, T, F]
        has_ant = True

    num_ut = tf.shape(h)[0]
    num_bs = tf.shape(h)[2]

    slope_ut_bs = tf.TensorArray(tf.float32, size=num_ut*num_bs)
    fD_ut_bs    = tf.TensorArray(tf.float32, size=num_ut*num_bs)

    idx = 0
    for ut in tf.range(num_ut):
        for bs in tf.range(num_bs):
            slopes_f = []
            fDs_f    = []
            for f0 in f_indices:
                if has_ant:
                    # promedio sobre antenas si no se promedió antes
                    uaN = tf.shape(h)[1]; baN = tf.shape(h)[3]
                    s_list = []; f_list = []
                    for ua in tf.range(uaN):
                        for ba in tf.range(baN):
                            x_t = H[ut, ua, bs, ba, :, f0]
                            s, fd = _slope_and_fd_for_one_f(x_t, Tsym)
                            s_list.append(s); f_list.append(fd)
                    slope = tf.reduce_mean(tf.stack(s_list))
                    fD    = tf.reduce_mean(tf.stack(f_list))
                else:
                    x_t = H[ut, bs, :, f0]   # [T]
                    slope, fD = _slope_and_fd_for_one_f(x_t, Tsym)

                slopes_f.append(slope)
                fDs_f.append(fD)

            slopes_f = tf.stack(slopes_f)  # [K]
            fDs_f    = tf.stack(fDs_f)     # [K]

            slope_agg = _median_1d(slopes_f) if use_median_over_f else tf.reduce_mean(slopes_f)
            fD_agg    = _median_1d(fDs_f)    if use_median_over_f else tf.reduce_mean(fDs_f)

            slope_ut_bs = slope_ut_bs.write(idx, slope_agg)
            fD_ut_bs    = fD_ut_bs.write(idx,    fD_agg)
            idx += 1

    slope_ut_bs = tf.reshape(slope_ut_bs.stack(), [num_ut, num_bs])   # [UT, BS]
    fD_ut_bs    = tf.reshape(fD_ut_bs.stack(),    [num_ut, num_bs])   # [UT, BS]
    nu_ut_bs    = fD_ut_bs / scs                                      # [UT, BS]
    Tc_ut_bs    = tf.where(tf.abs(fD_ut_bs) > 1e-9,
                           tf.constant(0.423, tf.float32)/tf.abs(fD_ut_bs),
                           tf.constant(1e9, tf.float32))

    return {
        "slope_rad_per_sym_ut_bs": slope_ut_bs,  # [UT, BS]
        "fD_est_hz_ut_bs":         fD_ut_bs,     # [UT, BS]
        "nu_ut_bs":                nu_ut_bs,     # [UT, BS]
        "Tc_seconds_ut_bs":        Tc_ut_bs,     # [UT, BS]
        "f_indices":               tf.convert_to_tensor(f_indices, tf.int32),
    }




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
                 frequency_mhz: float = 7000.0,   # Frecuencia portadora [MHz] #7000 doppler
                 tx_power_dbm: float = 20.0,      # Potencia TOTAL objetivo [dBm] (se reparte si hay sectores) #8 doppler

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
                 rx_array_polarization: str = "H",

                 # --- pose inicial del transmisor ---
                 tx_initial_position: tuple[float, float, float] = (0.0, 0.0, 10.0), # [m]
                 tx_orientation_deg: tuple[float, float, float] = (0.0, -90.0, 0.0), # [°] yaw,pitch,roll

                 # --- control del trazador de caminos (PathSolver) ---
                 max_depth: int = 5,              # Nº máx. de interacciones por camino
                 los: bool = True,                # Considerar Line-of-Sight
                 specular_reflection: bool = True,# Reflexiones especulares (reflexiones tipo espejo)
                 diffuse_reflection: bool = True, # Reflexiones difusas, por superficies rugosas (muy costoso, realista)
                 refraction: bool = True,         # Refracción (atravesar vidrios, etc. cambiar angulo y atenuar)
                 diffraction: bool = False,  # Difracción, activador general
                 edge_diffraction: bool = False,  # Difracción en aristas y esquinas
                 diffraction_lit_region: bool = False,
                 synthetic_array: bool = False,   # True: matriz sintética (rápido); False: por elemento (en false realista)
                 samples_per_src: int = 500_000,    # Nº de rayos por fuente (default 1,000,000)
                 max_num_paths_per_src: int = 500_000,  # Tope de caminos por fuente (None => default) (default 1000000)
                 seed: int = 41,                   # Semilla del muestreo estocástico



                # --- parametros SYS
                num_ut: int = 6,                # número de usuarios/receptores
                num_subcarriers: int = 128,     # número de subportadoras  #1024 doppler
                num_ofdm_symbols: int =12,     # número de símbolos OFDM   # 168 doppler
                bler_target: float = 0.1,       # objetivo de BLER para el enlace
                mcs_table_index: int = 1,       # índice de la tabla MCS a utilizar
                num_ut_ant: int = 1,            # número de antenas por usuario
                num_bs: int = 1,                # número de estaciones base                
                subcarrier_spacing: float = 30_000, #30e3 # Separación entre subportadoras [Hz] #7500 doppler
                temperatura: int = 294,        # temperatura en K (para cálculo de ruido térmico) 21°C = 294K



                #num_ut: int = None,     # si ya lo recibes, mantenlo; si no, se infiere más abajo
                doppler_enabled: bool = False,           # bandera global para activar/desactivar Doppler
                drone_velocity_mps: tuple[float, float, float] = (0.0, 0.0, 0.0),  # vx, vy, vz del dron en m/s
                rx_velocities_mps: list[tuple[float, float, float]] | None = None, 
               ):

        # --- Modo ---
        self.antenna_mode = str(antenna_mode).upper()

        # --- RF / ruido ---
        self.freq_hz = frequency_mhz * 1e6
        self.tx_power_dbm_total = tx_power_dbm

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
        self.diffraction = diffraction
        self.edge_diffraction = edge_diffraction
        self.diffraction_lit_region = diffraction_lit_region
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
        self.temperatura = temperatura


        #efecto doppler
        self.doppler_enabled = doppler_enabled
        self.drone_velocity_mps = drone_velocity_mps
        self.rx_velocities_mps = rx_velocities_mps
        self.tx_velocities = drone_velocity_mps


        # PHY Abstraction
        self.phy_abs = PHYAbstraction()

        # OLLA Link Adaptation (Outer Loop)
        self.olla = OuterLoopLinkAdaptation(
            self.phy_abs,
            num_ut=self.num_ut,
            bler_target=self.bler_target,
            batch_size=[self.num_bs]
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
            num_rows=self.tx_array_rows, 
            num_cols=self.tx_array_cols,
            vertical_spacing=self.tx_array_v_spacing,
            horizontal_spacing=self.tx_array_h_spacing,
            pattern=tx_pattern, polarization=tx_pol
        )
        self.scene.rx_array = PlanarArray(
            num_rows=self.rx_array_rows, 
            num_cols=self.rx_array_cols,
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


        # Crea el único TX
        tx = Transmitter(
            name="tx0",
            position=list(self.tx_initial_position),
            display_radius=2
        )
        tx.orientation = [_norm_deg(base_yaw), float(base_pitch), float(base_roll)]
        tx.power_dbm = float(self.tx_power_dbm_total)
        tx.velocity = self.tx_velocities
        #tx.velocity = [0,0,0]

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
                          display_radius=1.5, color=(0, 0, 0),
                          velocity = [0, 0, 0]
                          )
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
            diffraction=self.diffraction,
            edge_diffraction=self.edge_diffraction,
            diffraction_lit_region=self.diffraction_lit_region,
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
                rm = rm_solver(scene=self.scene,
                               max_depth=self.max_depth,
                               cell_size=[1, 1],
                               los=self.los,
                               samples_per_tx=self.max_num_paths_per_src,
                               specular_reflection=self.specular_reflection,
                               diffuse_reflection=self.diffuse_reflection,
                               refraction=self.refraction,
                               diffraction=self.diffraction,
                               edge_diffraction=self.edge_diffraction,
                               diffraction_lit_region=self.diffraction_lit_region
                               )
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
 

        temperatura = tf.constant(self.temperatura, tf.float32)
        subcarrier_spacing = tf.constant(self.subcarrier_spacing, tf.float32)

        no = BOLTZMANN_CONSTANT * temperatura * subcarrier_spacing  # Watts por subportadora
        EPS_NO = tf.constant(1e-18, tf.float32)
        no = tf.maximum(no, EPS_NO)

        # --- DOPPLER por-UT (y BS) ---
        metrics_Doppler = doppler_metrics_multi(
            h,
            ofdm_symbol_duration_s=float(self.resource_grid.ofdm_symbol_duration),
            scs_hz=float(self.subcarrier_spacing),
            f_indices=(5, 20, 40, 60, 80, 100),
            avg_over_antennas=True,
            use_median_over_f=True,        
        )

        # --- Ganancia de canal y tasa Shannon estimada (para scheduler PF) ---
        # h: [num_ut, num_ut_ant, num_bs, num_bs_ant, T, F]
        channel_gain = tf.maximum(tf.math.square(tf.abs(h)), 1e-12)   

        # log2(1 + SNR) por RE/antena
        rate = log2(1.0 + channel_gain / no)  # [ut, ut_ant, bs, bs_ant, T, F]

        # Promedio sobre antenas de UT y BS (¡solo antenas!)
        rate = tf.reduce_mean(rate, axis=[1, 3])  # -> [num_ut, num_bs, T, F]

        # Reordenar a lo que espera el scheduler
        rate_achievable_est = tf.transpose(rate, [1, 2, 3, 0])  # -> [num_bs, T, F, num_ut]

        

        # REs asignados por BS y UT: [num_bs, num_ut]
        allocation_mask = self._build_ofdma_equal_mask_rr_tf()

        num_allocated_re = tf.reduce_sum(tf.cast(allocation_mask, tf.int32), axis=[ 1, 2, 4])

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
            guaranteed_power_ratio=tf.maximum(0.5, 1.0/tf.cast(self.num_ut, tf.float32)),
            fairness=0
        )
        tx_power_per_ut = tf.nn.relu(tx_power_per_ut)  # no negativos
        

        # --- Reparto de potencia en REs asignados ---
        tx_power = spread_across_subcarriers(
            tf.expand_dims(tx_power_per_ut, axis=-2),  # [num_bs, num_ut, 1]
            allocation_mask,
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
            step_blocks, harq_feedback_masked, metrics_Doppler
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
        harq_feedback_masked,
        metrics_Doppler
        ) = self.sys_step(
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

       
        # ---- Doppler por-UE (BS=0) para guardar en info/plots ----
        fD_ut_bs    = metrics_Doppler["fD_est_hz_ut_bs"].numpy()[:, 0]          # [num_ut]
        slope_ut_bs = metrics_Doppler["slope_rad_per_sym_ut_bs"].numpy()[:, 0]  # [num_ut]
        nu_ut_bs    = metrics_Doppler["nu_ut_bs"].numpy()[:, 0]                 # [num_ut]
        Tc_ut_bs    = metrics_Doppler["Tc_seconds_ut_bs"].numpy()[:, 0]         # [num_ut]


        # ---------- Métricas por-UE ----------
        ue_metrics = []
        prx_list = self.compute_prx_dbm()
        prx_theo_list = self.compute_prx_dbm_theoretical( 
            gamma=getattr(self, "pathloss_gamma", 1.8),
            d0=1.0,
            Gt_dBi=getattr(self, "tx_gain_dbi", 0.0),
            Gr_dBi=getattr(self, "rx_gain_dbi", 0.0),
        )

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
                "prx_dbm_theo": float(prx_theo_list[i]), 
                "se_la": float(se_la_np[i]), #
                "se_shannon": float(se_shannon_np[i]), #
                "se_gap_pct": se_gap_pct_i, #

                # Métrica TBLER del step por UE (0/1/NaN si no transmitió)
                "tbler": float(tbler_np[i]), #

                # --- NUEVO: DOPPLER por UE ---
                "doppler_fd_hz": float(fD_ut_bs[i]),
                "doppler_slope_rad_per_sym": float(slope_ut_bs[i]),
                "doppler_nu_fd_over_scs": float(nu_ut_bs[i]),
                "doppler_Tc_seconds": float(Tc_ut_bs[i]),
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


    def _build_ofdma_equal_mask_rr_tf(self):
        """
        OFDMA Equal Frequency Partitioner (RR intra-step).
        Devuelve una máscara booleana de asignación de REs:
            shape: [num_bs, T, F, num_ut, num_streams_per_ut]
        Propósito:
        - Particionar F subportadoras equitativamente entre U UEs en cada símbolo t.
        - Repartir el residuo (F % U) dentro del mismo step mediante Round-Robin por símbolo.
        - Activar exclusivamente el stream 0 para mantener ortogonalidad estricta (sin MU-MIMO).

        """
        # Alias locales en int32
        num_bs = tf.cast(self.num_bs, tf.int32)
        T      = tf.cast(self.num_ofdm_symbols, tf.int32)
        F      = tf.cast(self.num_subcarriers, tf.int32)
        U      = tf.cast(self.num_ut, tf.int32)

        # Cálculo base y residuo
        base = F // U          # subportadoras por UE (parte entera)
        rem  = F %  U          # cuántos UEs reciben +1 en este símbolo

        # order_mat[t, k] = índice de UE que recibe el k-ésimo bloque en el símbolo t
        t_range   = tf.range(T, dtype=tf.int32)                           # [T]
        u_range   = tf.range(U, dtype=tf.int32)                           # [U]
        order_mat = (tf.expand_dims(t_range, 1) + tf.expand_dims(u_range, 0)) % U  # [T, U]

        # Índices de subportadora
        f_range = tf.range(F, dtype=tf.int32)                             # [F]
        cut     = (base + 1) * rem                                        # primer tramo (rem bloques de tamaño base+1)

        # Tramos grande/pequeño
        mask_big   = f_range <  cut                                        # [F] bool
        mask_small = tf.logical_not(mask_big)                              # [F] bool

        # Índices de bloque dentro de cada tramo
        # Evita /0 cuando base==0 con tf.maximum(base,1)
        block_idx_big   = tf.where(mask_big,  f_range // tf.maximum(base + 1, 1), tf.zeros_like(f_range))
        block_idx_small = tf.where(mask_small,(f_range - cut) // tf.maximum(base, 1), tf.zeros_like(f_range))

        # Mapear (t,f) -> UE usando order_mat
        # ue_big = order_mat[t, block_idx_big[f]]
        ue_big   = tf.gather(order_mat, block_idx_big, axis=1)             # [T, F]
        # ue_small = order_mat[t, rem + block_idx_small[f]]
        ue_small = tf.gather(order_mat, rem + block_idx_small, axis=1)     # [T, F]

        # Selección por tramo con broadcasting de la máscara [F] -> [1,F] sobre T
        ue_idx_t_f = tf.where(tf.expand_dims(mask_big, 0), ue_big, ue_small)  # [T, F]

        # One-hot numérico y cast a bool
        onehot_ue_num = tf.one_hot(ue_idx_t_f, depth=U, dtype=tf.int32)    # [T, F, U], int32
        onehot_ue = tf.cast(onehot_ue_num, tf.bool)                        # [T, F, U], bool

        # Expandir a [num_bs, T, F, U]
        onehot_ue = tf.expand_dims(onehot_ue, axis=0)                      # [1, T, F, U]
        onehot_ue = tf.tile(onehot_ue, [num_bs, 1, 1, 1])                  # [num_bs, T, F, U]

        # Activar solo stream 0
        num_streams = tf.cast(self.num_ut_ant, tf.int32)                   # usa tu valor actual
        stream0 = tf.one_hot(0, depth=num_streams, dtype=tf.int32)         # [S] numérico
        stream0 = tf.cast(stream0, tf.bool)                                # [S] bool
        stream0 = tf.reshape(stream0, [1, 1, 1, 1, num_streams])           # [1,1,1,1,S]
        stream0 = tf.tile(stream0, [num_bs, T, F, U, 1])                   # [num_bs,T,F,U,S]

        allocation_mask = tf.expand_dims(onehot_ue, axis=-1) & stream0        # [num_bs, T, F, U, S] bool
        return allocation_mask

    def set_velocities(
        self,
        doppler_enabled: bool,
        drone_velocity_mps: tuple[float, float, float],
        rx_velocities_mps: list[tuple[float, float, float]],
    ) -> None:
        """
        Copia las velocidades (m/s) a los objetos de la escena RT.
        - NO cambia posiciones.
        - Si doppler_enabled=False, fuerza v=(0,0,0) (Doppler=0).
        """
        
        if self.scene is None:
            return  # aún no construida

        # TX principal (asumo 1 dron -> 1 TX 'activo')
        try:
            vtx = (0.0, 0.0, 0.0) if not doppler_enabled else drone_velocity_mps
            if hasattr(self, "txs") and self.txs:
                # lista de TX (p.ej., 1 o 3 sectores). Aplica misma v a todos los sectores del dron
                for tx in self.txs:
                    tx_velocities = [float(vtx[0]), float(vtx[1]), float(vtx[2])]
            elif hasattr(self.scene, "tx") and self.scene.tx is not None:
                self.scene.tx_velocities = [float(vtx[0]), float(vtx[1]), float(vtx[2])]
        except Exception as e:
            print("[WARN] set_velocities: no se pudo setear TX.velocity:", e)

        # RX list (asumo que construiste self.rx_list)
        try:
            if hasattr(self, "rx_list") and self.rx_list:
                N = min(len(self.rx_list), len(rx_velocities_mps))
                for i in range(N):
                    vrx = (0.0, 0.0, 0.0) if not doppler_enabled else rx_velocities_mps[i]
                    self.rx_list[i].velocity = [float(vrx[0]), float(vrx[1]), float(vrx[2])]
        except Exception as e:
            print("[WARN] set_velocities: no se pudo setear RX[i].velocity:", e)






    # ---- Métricas adicionales Calculo teorico de Pr----


    # --- PRx teórico (Goldsmith 2.40): Pt_dBm + K_dB - 10*gamma*log10(d/d0) ---
    def compute_prx_dbm_theoretical(self,
                                    gamma: float = None,
                                    d0: float = 1.0,
                                    Gt_dBi: float = None,
                                    Gr_dBi: float = None) -> np.ndarray:
        """
        PRx teórico (modelo log-distancia):
            PRx[dBm] = Pt[dBm] + K[dB] - 10*γ*log10(d/d0)
        con K[dB] = 20*log10(λ/(4π d0)) + Gt + Gr
        """
        # parámetros por defecto desde el sistema si existen
        if gamma is None:
            gamma = getattr(self, "pathloss_gamma", 2.0)
        if Gt_dBi is None:
            Gt_dBi = float(getattr(self, "tx_gain_dbi", 0.0))
        if Gr_dBi is None:
            Gr_dBi = float(getattr(self, "rx_gain_dbi", 0.0))

        c = 299_792_458.0
        lam = c / float(self.freq_hz)

        K_dB = 20.0 * math.log10(lam / (4.0 * math.pi * d0)) + Gt_dBi + Gr_dBi
        Pt_dBm = float(self._total_tx_power_dbm())

        # posiciones actuales
        tx = np.array(self.txs[0].position, dtype=float).reshape(3)
        rx = np.array([list(r.position) for r in self.rx_list], dtype=float).reshape(-1, 3)
        d = np.linalg.norm(rx - tx, axis=1)

        ratio = np.maximum(d / float(d0), 1e-12)  # evita log10(0)
        prx_dbm = Pt_dBm + K_dB - 10.0 * float(gamma) * np.log10(ratio)
        return np.asarray(prx_dbm, dtype=float).reshape(-1)
    




    # validar si un movimiento A->B es válido (sin colisiones)
    @staticmethod
    def _np3(p):
        import numpy as np
        a = np.asarray(p, dtype=float).reshape(-1)
        if a.size != 3:
            raise ValueError("Se esperaban 3 componentes [x,y,z].")
        return a

    def is_move_valid(
        self,
        a, b,
        radius: float = 0.30,   # radio del dron (m)
        n_offsets: int = 12,    # muestreo lateral alrededor del eje (0 = solo línea central)
        eps: float = 1e-3,      # margen numérico para evitar autointersección
        check_bounds: bool = True
    ) -> bool:
        """
        Devuelve True si el tramo A->B está libre de colisiones (considerando un "tubo" de radio 'radius').
        Devuelve False si hay intersección con la geometría o si B está fuera de bounds (si check_bounds=True).
        """
        import numpy as np
        import mitsuba as mi
        import drjit as dr

        if self.scene is None:
            raise RuntimeError("SionnaRT: scene no está construida. Llama build_scene() antes.")

        a = self._np3(a); b = self._np3(b)
        d = b - a
        L = float(np.linalg.norm(d))
        if L <= 1e-9:
            return True  # no hay movimiento efectivo

        # Chequeo de límites (si se pide)
        if check_bounds:
            if getattr(self, "scene_bounds", None) is not None:
                pmin, pmax = self.scene_bounds
            else:
                pmin, pmax = self.scene_bounds_xyz()
            pmin = np.asarray(pmin, dtype=float); pmax = np.asarray(pmax, dtype=float)
            if np.any(b < (pmin - 1e-6)) or np.any(b > (pmax + 1e-6)):
                return False

        # Convertir a tipos Mitsuba
        a_mi = mi.Point3f(float(a[0]), float(a[1]), float(a[2]))
        b_mi = mi.Point3f(float(b[0]), float(b[1]), float(b[2]))
        dirv  = b_mi - a_mi
        L_mi  = dr.norm(dirv)
        dirv  = dirv / L_mi

        # Base ortonormal perpendicular a la dirección
        up = mi.Vector3f(0.0, 0.0, 1.0)
        n1 = dr.normalize(dr.cross(dirv, up))
        # Si casi paralelo a Z, usar otra referencia
        n1 = dr.select(dr.norm(n1) < 1e-6, dr.normalize(dr.cross(dirv, mi.Vector3f(0, 1, 0))), n1)
        n2 = dr.normalize(dr.cross(dirv, n1))

        # Offsets circulares para aproximar el radio del dron
        offsets = [mi.Vector3f(0.0, 0.0, 0.0)]
        if radius > 0.0 and n_offsets > 0:
            for k in range(int(n_offsets)):
                th = 2.0 * np.pi * (k / n_offsets)
                offsets.append(radius * np.cos(th) * n1 + radius * np.sin(th) * n2)

        mi_scene = self.scene.mi_scene
        L_lim = float(L) - eps

        # Si cualquier rayo choca, retornamos False
        for off in offsets:
            o = a_mi + off + eps * dirv
            ray = mi.Ray3f(o, dirv)
            ray.maxt = L_lim
            if mi_scene.ray_test(ray):
                return False

        # Ningún rayo intersectó
        return True

