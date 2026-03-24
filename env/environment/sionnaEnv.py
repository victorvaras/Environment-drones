#importaciones
from __future__ import annotations
import math
import numpy as np
from pathlib import Path
import math

#Importaciones Sionna RT
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

#Importaciones Sionna SYS
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

#Importaciones para el canal de comunicación
from sionna.phy.ofdm import ResourceGrid, RZFPrecodedChannel, LMMSEPostEqualizationSINR
from sionna.phy.mimo import StreamManagement
from sionna.sys.utils import spread_across_subcarriers
from sionna.rt import subcarrier_frequencies

# -------------------------------- Utilidades de la escena  --------------------------------
def _resolve_scene_path(scene_id: str) -> str | None:
    """
    Resuelve la ubicación física de los archivos de escenario o escena (.xml o .glb).
    
    Busca de forma jerárquica en el repositorio para garantizar la portabilidad
    del simulador entre distintos entornos de trabajo.
    """
    from pathlib import Path
    p = Path(scene_id)
    if p.exists():
        return str(p)

    #Prioridad 1: Directorio de escenas estándar de Sionna
    base = Path(__file__).resolve().parents[1] / "scenes"
    cand_xml = base / f"{scene_id}.xml"
    if cand_xml.exists():
        return str(cand_xml)
    cand_dir = base / scene_id
    if cand_dir.is_dir():
        xml = cand_dir / "scene.xml"
        if xml.exists():
            return str(xml)

    #Prioridad 2: Mapas de prueba específicos del proyecto
    base_maps = Path(__file__).resolve().parents[2] / "Mapas-Sionna"
    cand_file = base_maps / scene_id
    if cand_file.exists():
        return str(cand_file)
    cand_glb = base_maps / f"{scene_id}.glb"
    if cand_glb.exists():
        return str(cand_glb)

    return None


def load_builtin_scene(name: str = "munich",
                       frequency_hz: float = 3.5e9,
                       merge_shapes: bool = True):
    """
    Carga un entorno preconstruido en Sionna.

    Inicializa arreglos PlanarArray para Transmisores (TX) y  receptores (RX),
    definiendo la base para la propagación de rayos.
    """
    scene_obj = None
    if hasattr(builtin_scenes, name):
        scene_obj = getattr(builtin_scenes, name)
    scene = load_scene(scene_obj if scene_obj is not None else name,
                       merge_shapes=merge_shapes)
    scene.frequency = frequency_hz

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

# -------------------------------- Algoritmos de Estimación Doppler --------------------------------
def tf_unwrap_phase(phi: tf.Tensor) -> tf.Tensor:
    """
    Realiza el 'unwrapping' (desenrollado) de la fase para evitar discontinuidades.

    Paso matemático crítico para permitir una regresión lineal sobre la fase
    y estimar de manera continua el desplazamiento de frecuencia a lo largo del tiempo.
    """
    two_pi = tf.constant(2.0*np.pi, dtype=phi.dtype)
    d = phi[1:] - phi[:-1]
    d_adj = d - two_pi * tf.round(d / two_pi) #Corrección de saltos artificiales
    phi0 = phi[0:1]
    phi_tail = phi0 + tf.cumsum(d_adj)
    return tf.concat([phi0, phi_tail], axis=0)

def _slope_and_fd_for_one_f(x_t: tf.Tensor, Tsym: tf.Tensor):
    """
    Calcula la pendiente de la fase (rad/símbolo) y el Doppler instantáneo (fD)
    para una única subportadora.
    """
    phi = tf.math.angle(x_t)
    phi_unw = tf_unwrap_phase(phi)
    T = tf.shape(phi_unw)[0]
    slope = (phi_unw[-1] - phi_unw[0]) / tf.cast(tf.maximum(1, T-1), tf.float32)  #rad/símbolo
    two_pi = tf.constant(2.0*np.pi, tf.float32)
    fD = slope / (two_pi * Tsym)  #Hz (Doopler)
    return slope, fD

def _median_1d(x: tf.Tensor):
    """Calcula la mediana en tensores 1D para filtrar valores atípicos."""
    x_sorted = tf.sort(x, axis=0)
    n = tf.shape(x_sorted)[0]
    mid = n // 2
    return tf.cond(
        tf.equal(n % 2, 1),
        lambda: x_sorted[mid],
        lambda: 0.5*(x_sorted[mid-1] + x_sorted[mid])
    )

def doppler_metrics_multi(
    h: tf.Tensor,                       #Respuesta del canal [UT, Uant, BS, Bant, T, F]
    ofdm_symbol_duration_s: float,      #Duración del símbolo OFDM ($T_{sym}$) en seg
    scs_hz: float,                      #Espaciado entre subportadoras (SCS) en Hz
    f_indices: tuple = (5, 20, 40, 60, 80, 100),
    avg_over_antennas: bool = True,     #Promediar sobre Uant/Bant
    use_median_over_f: bool = True,     #Mediana sobre subportadoras elegidas
):
    """
    Calcula métricas de movilidad (Efecto Doppler) a partir de la matriz de canal h.

    Estima la tasa de cambio temporal de la fase entre símbolos OFDM para
    deducir la frecuencia Doppler (fD). Utiliza un enfoque estadístico
    (promedios o medianas) sobre múltiples subportadoras (f_indices) para
    mayor robustez frente al desvanecimiento selectivo en frecuencia.

    Returns:
        Un diccionario (Dict) con las siguientes métricas clave:
        - fD_est_hz_ut_bs: Desplazamiento Doppler estimado (Hz).
        - nu_ut_bs: Doppler normalizado respecto al SCS.
        - Tc_seconds_ut_bs: Tiempo de Coherencia (Tc) del canal estimado.
    """
    Tsym = tf.convert_to_tensor(ofdm_symbol_duration_s, tf.float32)
    scs  = tf.convert_to_tensor(scs_hz, tf.float32)

    #1.Agrupación espacial
    #Se reduce la dimensión de antenas (MIMO) promediando la respuesta del canal
    #para obtener una métrica representativa del enlace UT-BS.
    if avg_over_antennas:
        H = tf.reduce_mean(h, axis=(1,3))  #[UT, BS, T, F]
        has_ant = False
    else:
        H = h                              #[UT, Uant, BS, Bant, T, F]
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
                    #Promedio sobre antenas en caso de no promediarse antes
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

            slopes_f = tf.stack(slopes_f)  #[K]
            fDs_f    = tf.stack(fDs_f)     #[K]

            #2. Agregación en Frecuencia
            #Se usa la mediana para descartar subportadoras en desvanecimientos profundos
            slope_agg = _median_1d(slopes_f) if use_median_over_f else tf.reduce_mean(slopes_f)
            fD_agg    = _median_1d(fDs_f)    if use_median_over_f else tf.reduce_mean(fDs_f)

            slope_ut_bs = slope_ut_bs.write(idx, slope_agg)
            fD_ut_bs    = fD_ut_bs.write(idx,    fD_agg)
            idx += 1

    slope_ut_bs = tf.reshape(slope_ut_bs.stack(), [num_ut, num_bs])   #[UT, BS]
    fD_ut_bs    = tf.reshape(fD_ut_bs.stack(),    [num_ut, num_bs])   #[UT, BS]
    nu_ut_bs    = fD_ut_bs / scs                                             #[UT, BS]

    #3.Cálculo del Tiempo de Coherencia (Tc)
    #Basado en la aproximación se define el intervalo temporal donde
    #el canal se considera altamente correlacionado.
    Tc_ut_bs    = tf.where(tf.abs(fD_ut_bs) > 1e-9,
                           tf.constant(0.423, tf.float32)/tf.abs(fD_ut_bs),
                           tf.constant(1e9, tf.float32))

    return {
        "slope_rad_per_sym_ut_bs": slope_ut_bs,  #[UT, BS]
        "fD_est_hz_ut_bs":         fD_ut_bs,     #[UT, BS]
        "nu_ut_bs":                nu_ut_bs,     #[UT, BS]
        "Tc_seconds_ut_bs":        Tc_ut_bs,     #[UT, BS]
        "f_indices":               tf.convert_to_tensor(f_indices, tf.int32),
    }

# -------------------------------- Wrapper Sionna RT --------------------------------
"""
Valores por default del PathSolver: {
    'samples_per_src': 1000000, 'max_num_paths_per_src': 1000000, 'synthetic_array': True, 'max_depth': 3,
    'los': True, 'specular_reflection': True, 'diffuse_reflection': False, 'refraction': True, 'seed': 42}
"""
class SionnaRT:
    """
    Gestión de la escena 3D y evaluación de métricas de red mediante Ray Tracing (Sionna RT).

    Esta clase actúa como el motor de propagación del entorno. Configura los  transmisores (TX)
    y receptores (TX), utiliza el algoritmo de PathSolver para obtener la respuesta al impulso del 
    canal (CIR), y abstrae la capa física (PHY) para calcular métricas de rendimiento 
    como SINR y Eficiencia Espectral.
    """
    def __init__(self,
                 # --- Configuración Topológica de Antenas ---
                 antenna_mode: str = "ISO",       #Modos de radiación: "ISO" (Isotrópico) o "SECTOR3_3GPP" (Trisectorial realista 5G)

                 # --- RF / ruido ---
                 frequency_mhz: float = 7000.0,   #Frecuencia portadora [MHz]. #7000 (7GHz) es utilizado para el efecto doppler
                 tx_power_dbm: float = 20.0,      #Potencia total de transmisión [dBm]. Se divide entre sectores si aplica. #8 es utilizado para el efecto doppler

                 # --- Entorno de Simulación (escena de simulación) ---
                 scene_name: str = "munich",

                 # --- Transmisor (Dron/BS/TX): Matriz de Antenas ---
                 tx_array_rows: int = 1,                #Nº de filas de la matriz TX
                 tx_array_cols: int = 1,                #Nº de columnas de la matriz TX
                 tx_array_v_spacing: float = 0.5,       #Separación vertical (en λ)
                 tx_array_h_spacing: float = 0.5,       #Separación horizontal (en λ)
                 tx_array_pattern: str = "tr38901",     #"iso","dipole","tr38901", etc.
                 tx_array_polarization: str = "VH",     #"V","H","VH" (dual cruzada)

                 # --- Receptores (UE/RX): Matriz de Antenas ---
                 rx_array_rows: int = 1,
                 rx_array_cols: int = 1,
                 rx_array_v_spacing: float = 0.5,
                 rx_array_h_spacing: float = 0.5,
                 rx_array_pattern: str = "iso",
                 rx_array_polarization: str = "V",

                 # --- Posición inicial del transmisor ---
                 tx_initial_position: tuple[float, float, float] = (0.0, 0.0, 10.0), #Coordenadas cartesianas [x,y,z] en metros.
                 tx_orientation_deg: tuple[float, float, float] = (0.0, -90.0, 0.0), #Ángulos de Euler [°] [yaw, pitch, roll]. Pitch = -90° apunta hacia el suelo.

                 # --- Control del Trazador de Rayos (PathSolver) ---
                 #Define la precisión electromagnética vs costo computacional.
                 max_depth: int = 5,                        #Nº máx. de interacciones por camino (rebotes máximos)
                 los: bool = True,                          #Line-of-Sight (considerar o no la linea de vista)
                 specular_reflection: bool = True,          #Reflexiones especulares (reflexiones tipo espejo)
                 diffuse_reflection: bool = True,           #Reflexiones difusas (superficies rugosas). Alto costo, alta precisión y realismo
                 refraction: bool = True,                   #Refracción (penetración de materiales dieléctricos. Ej: ventanas, muros)
                 diffraction: bool = True,                  #Difracción según Teoría Uniforme de Difracción (UTD)
                 edge_diffraction: bool = True,
                 diffraction_lit_region: bool = True,
                 synthetic_array: bool = False,             #Si es False, simula la respuesta de cada elemento del array independientemente (realista).
                 samples_per_src: int = 1_000_000,          #Nº de rayos por fuente (default = 1.000.000)
                 max_num_paths_per_src: int = 1_000_000,    #Tope de caminos por fuente (None => default, default = 1.000.000)
                 seed: int = 41,                            #Semilla

                # --- Parámetros de Sistema (Sionna SYS) ---
                num_ut: int = 6,                        #Número de usuarios/receptores
                num_subcarriers: int = 128,             #Número de subportadoras  #1024 es utilizado para el efecto doppler
                num_ofdm_symbols: int =12,              #Número de símbolos OFDM (símbolos por slot de transmisión) #168 es utilizado para el efecto doppler
                bler_target: float = 0.1,               #Block Error Rate objetivo (BLER) para el enlace. Umbral para OLLA.
                mcs_table_index: int = 1,               #Indice de la tabla MCS. Tabla de Modulación y Codificación (Modulation and Coding Scheme).
                num_ut_ant: int = 1,                    #Número de antenas por usuario (por UE)
                num_bs: int = 1,                        #Número de transmisores (estaciones base)
                subcarrier_spacing: float = 30_000,     #Separación entre subportadoras [Hz] (SCS, 30e3) #7500 es utilizado para el efecto doppler
                temperatura: int = 294,                 #Temperatura de ruido en Kelvin (usada para cálculo de ruido térmico), 21°C = 294K.

                # --- Efecto Doppler ---
                #num_ut: int = None,                                               #Si ya se recibe, mantenerlo; sino, se infiere más abajo
                doppler_enabled: bool = False,                                     #Bandera global para activar o desactivar Doppler
                drone_velocity_mps: tuple[float, float, float] = (0.0, 0.0, 0.0),  #Vector velocidad del dron en m/s [vx, vy, vz]
                rx_velocities_mps: list[tuple[float, float, float]] | None = None, #Vector velocidad de usuarios en m/s (UEs)
               ):

        # --- Modo de antena ---
        self.antenna_mode = str(antenna_mode).upper()

        # --- RF / ruido ---
        self.freq_hz = frequency_mhz * 1e6
        self.tx_power_dbm_total = tx_power_dbm

        # --- Asignación de escena ---
        self.scene_name = scene_name


        # --- Configuracion de antenas de transmisores (TX) ---
        self.tx_array_rows = tx_array_rows
        self.tx_array_cols = tx_array_cols
        self.tx_array_v_spacing = tx_array_v_spacing
        self.tx_array_h_spacing = tx_array_h_spacing
        self.tx_array_pattern = tx_array_pattern
        self.tx_array_polarization = tx_array_polarization

        # --- Configuración de antenas de receptores (RX) ---
        self.rx_array_rows = rx_array_rows
        self.rx_array_cols = rx_array_cols
        self.rx_array_v_spacing = rx_array_v_spacing
        self.rx_array_h_spacing = rx_array_h_spacing
        self.rx_array_pattern = rx_array_pattern
        self.rx_array_polarization = rx_array_polarization

        # --- Posición inicial del transmisor ---
        self.tx_initial_position = tx_initial_position
        self.tx_orientation_deg = tx_orientation_deg  #[yaw, pitch, roll] en grados [°]

        # --- Configuración del PathSolver ---
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

        # --- Configuración de objetos RT ---
        self.scene: Scene | None = None
        self._solver: PathSolver | None = None
        self.tx: Transmitter | None = None
        self.txs: list[Transmitter] = []    #Lista de transmisores
        self.rx_list: list[Receiver] = []   #Lista de receptores

        # --- Configuración de objetos SYS ---
        self.num_ut = num_ut
        self.num_subcarriers = num_subcarriers
        self.num_ofdm_symbols = num_ofdm_symbols
        self.bler_target = bler_target
        self.mcs_table_index = mcs_table_index
        self.num_ut_ant = num_ut_ant
        self.num_bs = num_bs
        self.subcarrier_spacing = subcarrier_spacing
        self.temperatura = temperatura

        # --- Configuración de Doopler ---
        self.doppler_enabled = doppler_enabled
        self.drone_velocity_mps = drone_velocity_mps
        self.rx_velocities_mps = rx_velocities_mps
        self.tx_velocities = drone_velocity_mps

        # -------------------------------- Abstracción de Capa Física (PHY) --------------------------------
        self.phy_abs = PHYAbstraction()

        #Adaptación de Enlace de Lazo Externo (Outer Loop Link Adaptation - OLLA)
        #Ajusta dinámicamente el SINR efectivo para cumplir con el `bler_target` (BLER objetivo).
        self.olla = OuterLoopLinkAdaptation(
            self.phy_abs,
            num_ut=self.num_ut,
            bler_target=self.bler_target,
            batch_size=[self.num_bs]
        )

        #Malla de Recursos OFDM (Resource Grid)
        #Define la estructura tiempo-frecuencia de la transmisión.
        self.resource_grid = ResourceGrid(
            num_ofdm_symbols=self.num_ofdm_symbols,
            fft_size=self.num_subcarriers,
            subcarrier_spacing=self.subcarrier_spacing,
            num_tx=self.num_ut,
            num_streams_per_tx=self.num_ut_ant
        )

        #Gestión de flujos espaciales para MIMO
        self.stream_management = StreamManagement(
            tf.ones([self.num_ut, self.num_bs]), self.num_ut * self.num_ut_ant
        )

        #Precodificación espacial
        #Mitiga la interferencia multiusuario (MUI) en el transmisor.
        self.precoded_channel = RZFPrecodedChannel(
            resource_grid=self.resource_grid,
            stream_management=self.stream_management
        )

        #Ecualización en el receptor
        #Estima el SINR post-ecualización asumiendo un receptor LMMSE (Linear Minimum Mean Square Error).
        self.lmmse_posteq_sinr = LMMSEPostEqualizationSINR(
            resource_grid=self.resource_grid,
            stream_management=self.stream_management
        )

        #Variables de estado para el ciclo de realimentación (Feedback)
        self.harq_feedback = -tf.ones([self.num_bs, self.num_ut], dtype=tf.int32)
        self.sinr_eff_feedback = tf.zeros([self.num_bs, self.num_ut], dtype=tf.float32)
        self.num_decoded_bits = tf.zeros([self.num_bs, self.num_ut], dtype=tf.int32)

    # ---- Construcción de Escena ----
    def build_scene(self):
        """Carga y construye el escenario para la simulación"""
        xml_path = _resolve_scene_path(self.scene_name)

        if xml_path is not None:
            if xml_path.endswith((".glb", ".gltf", ".obj")):
                #Escena externa
                scene = load_scene(xml_path, merge_shapes=True)
            else:
                #Escena XML estándar
                scene = load_scene(xml_path, merge_shapes=True)
        else:
            #Escena interna de Sionna
            scene, _ = load_builtin_scene(name=self.scene_name,
                                        frequency_hz=self.freq_hz,
                                        merge_shapes=True)

        self.scene = scene
        self.scene.frequency = self.freq_hz

        #Se asegura el acceso al motor de renderizado interno (Mitsuba)
        #Necesario para consultas de intersección directa independientes del PathSolver.
        if hasattr(self.scene, "mi_scene"):
            self.mi_scene = self.scene.mi_scene
        else:
            self.mi_scene = self.scene

        pmin, pmax = self.scene_bounds_xyz()

        #Se guardan los límites de la escena para que Gymnasium pueda utilizarlos
        self.scene_bounds = (pmin, pmax)

        #Se configuran los arrays globales de las antenas (se aplican a todos los TX/RX)
        tx_pattern = self.tx_array_pattern
        tx_pol = self.tx_array_polarization
        if self.antenna_mode in ("SECTOR3_3GPP", "SECTOR3", "3GPP"):
            if tx_pattern == "iso":
                tx_pattern = "tr38901"
            if tx_pol == "V":
                tx_pol = "VH"

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

        #Se desactivan la precodificación/combinación analógica por defecto
        #para manejar el procesamiento espacial enteramente en banda base digital
        if hasattr(self.scene, "transmit_precoder"):    
            self.scene.transmit_precoder = None
        if hasattr(self.scene, "receive_combiner"):
            self.scene.receive_combiner = None

        self._solver = PathSolver()
        self._create_transmitters()

        # Se verifica la integridad del motor de trazado
        assert self.scene is not None and self._solver is not None and self.tx is not None, \
            "Sionna RT no quedó inicializado correctamente."
    
    def _create_transmitters(self):
        """
        Instancia y configura la Estación Base (BS/Dron) en el motor RT.

        Independientemente del patrón de radiación (ISO o 3GPP), se inicializan los
        nodos transmisores. La orientación angular (Pitch/Yaw/Roll) define el área de
        cobertura.
        """
        #Se limpia lista local para evitar duplicación de referencias en la memoria de la escena
        self.txs = []

        def _norm_deg(a: float) -> float:
            """Normaliza ángulos al rango [0, 360) grados."""
            x = float(a) % 360.0
            return x if x >= 0.0 else x + 360.0

        #Orientación base [yaw, pitch, roll]
        try:
            base_yaw, base_pitch, base_roll = self.tx_orientation_deg
        except Exception as e:
            raise ValueError("tx_orientation_deg debe ser [yaw, pitch, roll] en grados") from e

        #Creación del nodo TX en Mitsuba/Sionna
        tx = Transmitter(
            name="tx0",
            position=list(self.tx_initial_position),
            display_radius=2
        )
        tx.orientation = [_norm_deg(base_yaw), float(base_pitch), float(base_roll)]
        tx.power_dbm = float(self.tx_power_dbm_total)

        #Asignación del vector velocidad (Dron) para el cálculo Doppler en el PathSolver
        tx.velocity = self.tx_velocities
        #tx.velocity = [0,0,0]

        try:
            tx.array = self._tx_array
            i=1
        except Exception:
            pass

        #Se añade a la escena y se guardan referencias
        self.scene.add(tx)
        self.txs.append(tx)
        self.tx = tx

    def remove_receivers(self):
        """
        Purga los receptores (UEs) de la escena para permitir un reset limpio.

        Esencial para la limpieza entre pasos (steps) o episodios
        de Gymnasium, previniendo colisiones de namespaces (ej. 'RX_0 already used')
        y fugas de memoria en el motor de renderizado.
        """
        if self.scene is None:
            return

        #1.Se identifican los receptores a borrar (los que comienzan con "RX_")
        #Se usa list() para crear una copia de las claves y no romper el iterador al borrar
        objs_to_remove = []
        try:
            #Se itera sobre los receptores que tiene Sionna en memoria
            for name in self.scene.receivers.keys():
                if name.startswith("RX_"):
                    objs_to_remove.append(name)

            #2.Se borran de la escena
            for name in objs_to_remove:
                self.scene.remove(name)

        except Exception as e:
            print(f"[SionnaRT] Advertencia al limpiar receptores: {e}")

        #3.Se limpia la lista interna de referencias
        #Importante para que no queden residuos en la lista
        self.rx_list = []

    def attach_receivers(self, rx_positions_xyz: np.ndarray):
        """
        Instancia los receptores (UEs) en base a las posiciones calculadas
        por el Gestor de Movilidad (SFM).
        """
        assert self.scene is not None, "build_scene() no fue llamado."
        self.rx_list = []
        for i, p in enumerate(rx_positions_xyz):
            #Se inicializan estáticos (velocity=[0,0,0]) respecto a la escena.
            #La velocidad real para el Doppler se inyecta posteriormente en el canal.
            rx = Receiver(name=f"RX_{i}",
                          position=[float(p[0]), float(p[1]), float(p[2])],
                          display_radius=1.5, color=(0, 0, 0),
                          velocity = [0, 0, 0]
                          )

            try:
                rx.array = self._rx_array
                i=1
            except Exception:
                pass

            self.scene.add(rx)
            self.rx_list.append(rx)

    def move_tx(self, pos_xyz, drone_velocity_mps):
        """
        Actualiza el estado (posición y velocidad) del UAV.
        Este metodo es invocado por Gymnasium en cada step() para reflejar la acción del dron.
        """
        assert self.txs, "TX no inicializados."
        pos = [float(pos_xyz[0]), float(pos_xyz[1]), float(pos_xyz[2])]
        #drone_velocity_mps =  (5.0, 5.0, 0.0)
        for tx in self.txs:
            tx.position = pos
            tx.velocity = drone_velocity_mps

    def scene_bounds_xyz(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Calcula el Bounding Box global de la escena en metros.
        Extrae las dimensiones espaciales directamente del motor Mitsuba.
        """
        mi_scene = self.scene.mi_scene            #Sionna -> Mitsuba - scene
        bb = mi_scene.bbox()                      #(min, max)

        pmin = np.array([float(bb.min.x), float(bb.min.y), float(bb.min.z)], dtype=float)

        #Se escala Z*2 para asegurar que el Bounding Box cubra el espacio aéreo del dron
        pmax = np.array([float(bb.max.x), float(bb.max.y), float(bb.max.z*2)], dtype=float)
        return pmin, pmax

    # ---- Cálculo de paths y métricas ----
    def _paths(self):
        """
        Lanza el motor de Ray Tracing (PathSolver).

        Calcula la propagación estocástica de los frentes de onda,
        considerando los fenómenos de reflexión, refracción y difracción configurados.
        Retorna los objetos "Paths" que contienen las trayectorias e información de fase.
        """
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
        """Suma la potencia de transmisión de todos los transmisores activos (en dBm)."""
        #Si existe lista de TX creada aún, se utiliza la potencia total configurada
        if not self.txs:
            return float(self.tx_power_dbm_total)

        p_mw = 0.0
        for tx in self.txs:
            try:
                p_dbm = float(tx.power_dbm) #power_dbm puede ser drjit.Float; lo convertimos a float de Python
            except Exception:
                p_dbm = float(self.tx_power_dbm_total)
            p_mw += 10.0 ** (p_dbm / 10.0)

        #Evita log10(0) y asegura tipo nativo
        p_mw = float(p_mw)
        return 10.0 * math.log10(p_mw if p_mw > 0.0 else 1e-30)

    def compute_prx_dbm(self) -> np.ndarray:
        """
        Estima la Potencia Recibida en dBm por receptor utilizando CIR de Sionna RT.

        Este metodo es una aproximación de gran escala . Extrae la
        amplitud compleja de la Respuesta al Impulso del Canal (CIR) de los rayos
        calculados, y suma la potencia lineal aportada por cada trayecto multipath.
        """
        paths = self._paths()
        a, _ = paths.cir(out_type="numpy")  #a: coeficiente de atenuación complejo
        a = np.atleast_1d(a)

        #Suma incoherente de la potencia de los multipaths
        if a.ndim == 0:
            power_lin = np.array([np.abs(a) ** 2], dtype=float)
        else:
            axes_to_sum = tuple(i for i in range(a.ndim) if i != 0)
            power_lin = np.sum(np.abs(a) ** 2, axis=axes_to_sum)

        #Piso mínimo matemático para evitar errores de logaritmo
        power_lin = np.maximum(power_lin.astype(float), 1e-24)
        ptx_dbm_total = self._total_tx_power_dbm()

        #Balance de Enlace simplificado
        prx_dbm = ptx_dbm_total + 10.0 * np.log10(power_lin)
        return prx_dbm.astype(float)

    # ---- Visualización opcional ----
    def preview_scene(self):
        """
        Abre un visor interactivo de la escena 3D.
        """
        assert self.scene is not None, "Scene no inicializada."
        try:
            self.scene.preview()
        except Exception as e:
            print("preview() requiere Jupyter. Usa render_scene_to_file().", e)

    def render_scene_to_file(self, filename: str = "scene.png",
                             resolution: tuple[int, int] = (900, 700), #900,700
                             with_radio_map: bool = False) -> bool:
        """
        Genera una representación visual estática (Render 2D) del estado actual de la simulación.
        """
        assert self.scene is not None, "Scene no inicializada."

        cam = self._auto_camera()

        #Posibilidad de aumentar tamaño de transmisor y receptor para visualización (en caso de escena grandes)
        aumentar = False
        if (aumentar):
            for rx in self.scene.receivers.values():
                rx.display_radius = 5
                #rx.color = (0, 0.9, 1)

            for tx in self.scene.transmitters.values():
                tx.display_radius = 5

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
        """
        Calcula dinámicamente la posición y enfoque óptimos de la cámara virtual.

        Utiliza el Bounding Box de la escena para asegurar que tanto la topología
        como el Dron y los receptores queden dentro del campo de visión (FOV)
        al momento de generar el render, usando una vista cenital/isométrica.
        """
        try:
            aabb = self.scene_bounds
            mn, mx = aabb[0], aabb[1]

            #Centro geométrico de la escena en XY
            cx = float((mn[0] + mx[0]) / 2)
            cy = float((mn[1] + mx[1]) / 2)

            #Dimensión máxima para calcular la altura requerida
            size_xy = max(float(mx[0] - mn[0]), float(mx[1] - mn[1]))

            #Se establece la cámara a una altura proporcional al tamaño de la escena
            z = max(150.0, size_xy * z_scale)
            return Camera(position=[cx, cy, z], look_at=[cx, cy, 0.0])
        except Exception:
            #Fallback de seguridad si falla la lectura del Bounding Box
            print("No se pudo calcular cámara automáticamente, usando fallback.")
            return Camera(position=[0, 0, 300], look_at=[0, 0, 0])

    # --- Cálculo de Sistema - Sionna SYS (Capa Física) ---
    @tf.function  # (jit_compile=True)
    def sys_step(self, h, harq_feedback, sinr_eff_feedback, num_decoded_bits):
        """
        Ejecuta un step de simulación del sistema usando Sionna SYS.

        Toma la Respuesta al Impulso del Canal y el estado del ciclo anterior para
        calcular la potencia de ruido térmico, aplicar control de potencia, estimar el
        SINR post-ecualización, adaptar el enlace (OLLA) y evaluar el éxito de los
        Bloques de Transporte (Transport Block Error Rate - TBLER).
        """
        temperatura = tf.constant(self.temperatura, tf.float32)
        subcarrier_spacing = tf.constant(self.subcarrier_spacing, tf.float32)

        #Cálculo del Piso de Ruido Térmico
        no = BOLTZMANN_CONSTANT * temperatura * subcarrier_spacing  #Watts por subportadora
        EPS_NO = tf.constant(1e-18, tf.float32)
        no = tf.maximum(no, EPS_NO)                                 #Piso matemático para evitar divergencias en cálculos de SNR

        # --- Estimación Doppler multi-subportadora ---
        metrics_Doppler = doppler_metrics_multi(
            h,
            ofdm_symbol_duration_s=float(self.resource_grid.ofdm_symbol_duration),
            scs_hz=float(self.subcarrier_spacing),
            f_indices=(5, 20, 40, 60, 80, 100),
            avg_over_antennas=True,
            use_median_over_f=True,        
        )

        # --- Ganancia del canal y Capacidad de Shannon ---
        #h: [num_ut, num_ut_ant, num_bs, num_bs_ant, T, F]
        channel_gain = tf.maximum(tf.math.square(tf.abs(h)), 1e-12)   

        #Tasa teórica máxima (Shannon Capacity): C = log2(1 + SNR) por Resource Element (RE)
        rate = log2(1.0 + channel_gain / no)  # [ut, ut_ant, bs, bs_ant, T, F]
        rate = tf.reduce_mean(rate, axis=[1, 3])  # -> [num_ut, num_bs, T, F]
        rate_achievable_est = tf.transpose(rate, [1, 2, 3, 0])

        # --- Asignación de Recursos OFDM (Scheduler) ---
        allocation_mask = self._build_ofdma_equal_mask_rr_tf()
        num_allocated_re = tf.reduce_sum(tf.cast(allocation_mask, tf.int32), axis=[ 1, 2, 4])

        # --- Pérdida de Trayecto Promedio (Pathloss) ---
        pathloss_per_ut = tf.reduce_mean(1.0 / channel_gain, axis=[1, 3, 4, 5])  #[num_ut, num_bs]
        pathloss_per_ut = tf.transpose(pathloss_per_ut, [1, 0])            #[num_bs, num_ut]

        pathloss_per_ut = tf.where(tf.math.is_finite(pathloss_per_ut),
                           pathloss_per_ut,
                           tf.reduce_max(pathloss_per_ut[tf.math.is_finite(pathloss_per_ut)]) )

        # --- Control de Potencia del Downlink (DL Power Control) ---
        #Asegura equidad (fairness) en la distribución de energía del dron hacia los UEs,
        #compensando parcialmente el Pathloss.
        tx_power_per_ut, _ = downlink_fair_power_control(
            pathloss_per_ut, no, num_allocated_re,
            bs_max_power_dbm=self.tx_power_dbm_total,
            guaranteed_power_ratio=tf.maximum(0.5, 1.0/tf.cast(self.num_ut, tf.float32)),
            fairness=0 #0 indica máxima eficiencia sistémica; 1 indicaría máxima equidad.
        )
        tx_power_per_ut = tf.nn.relu(tx_power_per_ut)

        # --- Reparto de potencia sobre la malla OFDM ---
        tx_power = spread_across_subcarriers(
            tf.expand_dims(tx_power_per_ut, axis=-2),  #[num_bs, num_ut, 1]
            allocation_mask,
            num_tx=self.num_bs
        )

        #Precodificación espacial RZF
        #Se introduce un factor de regularización proporcional al ruido
        #para evitar el mal acondicionamiento de la matriz de canal a bajas relaciones SNR.
        h_eff = self.precoded_channel(h[tf.newaxis, ...],
                                    tx_power=tx_power,
                                    alpha=no*20.0 + EPS_NO)

        #Recepción LMMSE para estimación de SINR
        sinr = self.lmmse_posteq_sinr(h_eff, no=no + EPS_NO, interference_whitening=False)
        #sinr: [num_bs, num_ofdm_symbols, num_subcarriers, num_ut, num_streams_per_ut]

        # --- Adaptación de Enlace (Outer Loop Link Adaptation - OLLA) ---
        #Determina dinámicamente el esquema de modulación y codificación (MCS) óptimo.
        mcs_index = self.olla(
            num_allocated_re=num_allocated_re,
            sinr_eff=sinr_eff_feedback,
            mcs_table_index=self.mcs_table_index,
            mcs_category=1,  # downlink
            harq_feedback=harq_feedback
        )

        # --- Abstracción de Capa Física (PHY Abstraction) ---
        #Simula la decodificación del Bloque de Transporte (TB) y devuelve el estado HARQ.
        num_decoded_bits, harq_feedback, sinr_eff_true, *_ = self.phy_abs(
            mcs_index, sinr=sinr, mcs_table_index=self.mcs_table_index, mcs_category=1
        )

        #Enmascarado HARQ: Si un UE no fue agendado en este step, su feedback es -1.
        # num_allocated_re: [num_bs, num_ut]
        harq_feedback_masked = tf.where(
            num_allocated_re > 0,                 #Programado en este step
            harq_feedback,                        #1=ACK / 0=NACK
            -tf.ones_like(harq_feedback)          #-1 si NO fue agendado (Unscheduled)
        )

        sinr_eff_db_true = lin_to_db(sinr_eff_true)  #Effective SINR

        #Feedback para OLLA en el próximo step (0 si no fue agendado)
        sinr_eff_feedback = tf.where(num_allocated_re > 0, sinr_eff_true, 0)

        # --- Evaluación de Eficiencia Espectral (Spectral Efficiency - SE) ---
        mod_order, coderate = decode_mcs_index(
            mcs_index, table_index=self.mcs_table_index, is_pusch=False
        )

        #Eficiencia Espectral real lograda por el Link Adaptation (SE-LA).
        #Solo aporta rendimiento si el paquete fue decodificado con éxito (ACK).
        se_la = tf.where(
            harq_feedback == 1,
            tf.cast(mod_order, coderate.dtype) * coderate,
            tf.cast(0, tf.float32)
        )
        se_shannon = log2(1.0 + sinr_eff_true)  #Cota teórica superior (SE-Shannon)

        # --- Evaluación TBLER (Transport Block Error Rate) ---
        # Indicador binario de intento de transmisión
        tb_tx_step_per_ue = tf.cast(tf.not_equal(harq_feedback[0, :], -1), tf.int32)

        # Éxito de transmisión (ACK = 1)
        tb_ok_step_per_ue = tf.cast(tf.equal(harq_feedback[0, :], 1), tf.int32) * tb_tx_step_per_ue

        #TBLER step: 0 (Éxito), 1 (Fallo), NaN (No agendado)
        tbler_step_per_ue = tf.where(
            tb_tx_step_per_ue > 0,
            1.0 - tf.cast(tb_ok_step_per_ue, tf.float32),
            tf.constant(float('nan'), dtype=tf.float32)
        )

        tbler_step_per_ue = tbler_step_per_ue[tf.newaxis, :]

        # --- Contadores de bloques para acumulados ---
        #Permiten TBLER acumulada = 1 - blocks_ok_accum / blocks_tx_accum
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
        """
        Recupera el tensor de la Respuesta en Frecuencia del Canal (CFR) calculado
        para el instante de simulación (step) actual.
        """
        if not hasattr(self, 'current_cfr_for_current_step'):
            raise ValueError("No se ha actualizado current_cfr_for_current_step aún.")
        return self.current_cfr_for_current_step

    def run_sys_step(self):
        """
        Ejecuta el ciclo completo de Capa Física y formatea las métricas.

        Este metodo actúa como puente entre el motor tensorial (TensorFlow) y la
        lógica de control (Python/NumPy). Actualiza el canal, ejecuta sys_step(),
        desempaqueta los resultados y construye diccionarios de métricas por usuario (UE)
        que serán consumidos por el entorno de Gymnasium para calcular la recompensa (Reward).
        """
        #Actualiza CFR para el step actual y obtiene tensor de canal
        self.update_and_store_cfr_for_step()
        h = self.get_current_channel_tensor()

        (self.harq_feedback,
        self.sinr_eff_feedback,
        self.num_decoded_bits,
        se_la,
        se_shannon,
        sinr_eff_db_true,
        tbler_per_user,
        step_blocks,
        harq_feedback_masked,
        metrics_Doppler
        ) = self.sys_step(
            h=h,
            harq_feedback=self.harq_feedback,
            sinr_eff_feedback=self.sinr_eff_feedback,
            num_decoded_bits=self.num_decoded_bits
        )

        # ---------- Inicialización de Acumuladores de Rendimiento ----------
        if not hasattr(self, "blocks_acc_tx"):
            self.blocks_acc_tx = [0 for _ in range(self.num_ut)]
        if not hasattr(self, "blocks_acc_ok"):
            self.blocks_acc_ok = [0 for _ in range(self.num_ut)]
        if not hasattr(self, "bits_acc_total"):
            self.bits_acc_total = [0 for _ in range(self.num_ut)]

        # ---------- Conversión Tensor a Numpy (Extracción de Métricas) ----------
        sinr_eff_db_true_np = sinr_eff_db_true.numpy()[0, :]  #[num_ut]
        se_la_np = se_la.numpy()[0, :]                        #[num_ut]
        se_shannon_np = se_shannon.numpy()[0, :]              #[num_ut]
        tbler_np = tbler_per_user.numpy()[0, :]               #[num_ut]  (0/1/NaN)
        bits_np = self.num_decoded_bits.numpy()[0, :]         #[num_ut]

        blocks_tx_step_per_ue_np = step_blocks["blocks_tx_step_per_ue"].numpy()  #[num_ut]
        blocks_ok_step_per_ue_np = step_blocks["blocks_ok_step_per_ue"].numpy()  #[num_ut]

        # ---------- Acumulación Histórica por UE ----------
        for i in range(self.num_ut):
            self.blocks_acc_tx[i] += int(blocks_tx_step_per_ue_np[i])
            self.blocks_acc_ok[i] += int(blocks_ok_step_per_ue_np[i])
            self.bits_acc_total[i] += int(bits_np[i])
       
        # ---- Extracción de Métricas Doppler por UE (Asumiendo BS=0) ----
        fD_ut_bs    = metrics_Doppler["fD_est_hz_ut_bs"].numpy()[:, 0]          #[num_ut]
        slope_ut_bs = metrics_Doppler["slope_rad_per_sym_ut_bs"].numpy()[:, 0]  #[num_ut]
        nu_ut_bs    = metrics_Doppler["nu_ut_bs"].numpy()[:, 0]                 #[num_ut]
        Tc_ut_bs    = metrics_Doppler["Tc_seconds_ut_bs"].numpy()[:, 0]         #[num_ut]

        # ---------- Construcción de Diccionario de Métricas por UE ----------
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

            #Cálculo de la brecha porcentual entre SE real y cota teórica de Shannon
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

                #Métrica TBLER del step por UE (0/1/NaN)
                "tbler": float(tbler_np[i]), #

                # --- Métricas Efecto Doppler --
                "doppler_fd_hz": float(fD_ut_bs[i]),
                "doppler_slope_rad_per_sym": float(slope_ut_bs[i]),
                "doppler_nu_fd_over_scs": float(nu_ut_bs[i]),
                "doppler_Tc_seconds": float(Tc_ut_bs[i]),
            })

        # ---------- Historial HARQ ----------
        if not hasattr(self, "harq_feedback_hist"):
            self.harq_feedback_hist = []  #Lista de arrays [num_bs, num_ut] con estados -1/0/1

        harq_mask_np = harq_feedback_masked.numpy()    #[num_bs, num_ut]
        self.harq_feedback_hist.append(harq_mask_np)   #Se guarda el estado del step actual

        #Construcción del tensor histórico: [Steps, num_bs, num_ut]
        harq_hist_np = np.stack(self.harq_feedback_hist, axis=0)

        #Evaluación sobre la Estación Base principal (BS=0, el Dron)
        harq_bs0 = harq_hist_np[:, 0, :]   #[Steps, num_ut]

        #Se reemplaza -1 (Unscheduled) por NaN para utilizar funciones ignorando NaNs
        #Esto evita penalizar el rendimiento por bloques de tiempo donde el UE no fue agendado.
        harq_bs0_nan = harq_bs0.astype(float)
        harq_bs0_nan[harq_bs0_nan == -1] = np.nan  #1=ACK, 0=NACK, NaN=Unscheduled

        #Acumuladores históricos de intentos (TX) y éxitos (OK)
        tx_cum = np.cumsum(~np.isnan(harq_bs0_nan), axis=0).astype(float)  #[Steps, num_ut]
        ok_cum = np.nancumsum(harq_bs0_nan, axis=0)                        #[Steps, num_ut]

        #Cálculo del TBLER Running
        tbler_running_per_ue = 1.0 - (ok_cum / np.maximum(tx_cum, 1.0))  #[Steps, num_ut]

        #Se extrae el valor actual (último step)
        tbler_running_per_ue_step = tbler_running_per_ue[-1, :]  # [num_ut]

        return {
            "ue_metrics": ue_metrics,
            "tbler_running_per_ue": tbler_running_per_ue_step.tolist(),            
        }

    def update_and_store_cfr_for_step(self):
        """
        Calcula la Respuesta en Frecuencia del Canal (CFR) para el step actual.

        Extrae los coeficientes complejos del canal a partir de los rayos trazados
        por el PathSolver, los evalúa en las frecuencias de subportadora OFDM
        configuradas, y almacena el tensor resultante con la dimensionalidad estricta
        requerida por el módulo Sionna SYS: [num_ut, num_ut_ant, num_bs, num_bs_ant, T, F].
        """
        #1) Obtención de trayectorias (Paths) y malla de frecuencias OFDM
        paths = self._paths()
        frequencies = subcarrier_frequencies(
            num_subcarriers=self.num_subcarriers,
            subcarrier_spacing=self.subcarrier_spacing
        )

        #2) Evaluación del CFR desde el motor RT (retorno en NumPy)
        #Dimensionalidad nativa: [num_rx, num_tx, T, F]
        h_freq_np = paths.cfr(
            frequencies=frequencies,
            sampling_frequency=1 / self.resource_grid.ofdm_symbol_duration,
            num_time_steps=self.num_ofdm_symbols,
            out_type="numpy"
        )

        # 3) Formateo a tensor complejo de TensorFlow (float32 real/imag -> complex64)
        h_tf = tf.convert_to_tensor(h_freq_np, dtype=tf.complex64)
        self.current_cfr_for_current_step = h_tf


    def _build_ofdma_equal_mask_rr_tf(self):
        """
        Generador de Máscaras de Asignación OFDMA (Round-Robin Equal Partitioner).
        Devuelve una máscara booleana tensorial (de asignación de REs):
        [num_bs, T, F, num_ut, num_streams_per_ut].

        Lógica de Diseño:
        - Particiona la banda (F subportadoras) equitativamente entre los UEs activos.
        - Si existe un residuo matemático (F % U != 0), lo distribuye equitativamente
          iterando sobre los símbolos OFDM (T) mediante un esquema Round-Robin.
        - Fuerza la ortogonalidad espacial estricta activando únicamente el stream 0,
          evitando transmisión MU-MIMO en la misma subportadora.
        """
        num_bs = tf.cast(self.num_bs, tf.int32)
        T      = tf.cast(self.num_ofdm_symbols, tf.int32)
        F      = tf.cast(self.num_subcarriers, tf.int32)
        U      = tf.cast(self.num_ut, tf.int32)

        #Cálculo de partición base y residuo
        base = F // U          #Subportadoras por UE (parte entera)
        rem  = F %  U          #Cuántos UEs reciben +1 en este símbolo

        #Generación de la matriz de turnos Round-Robin [T, U]
        t_range   = tf.range(T, dtype=tf.int32)   #[T]
        u_range   = tf.range(U, dtype=tf.int32)   #[U]
        order_mat = (tf.expand_dims(t_range, 1) + tf.expand_dims(u_range, 0)) % U  #[T, U]
        #order_mat[t, k] = índice de UE que recibe el k-ésimo bloque en el símbolo t

        #Lógica de índices de subportadora
        f_range = tf.range(F, dtype=tf.int32)   #[F]
        cut     = (base + 1) * rem              #Primer tramo (rem bloques de tamaño base+1)

        #Tramos big/small
        mask_big   = f_range <  cut             #[F] bool
        mask_small = tf.logical_not(mask_big)   #[F] bool

        # Índices de bloque dentro de cada tramo
        # Evita /0 cuando base==0 con tf.maximum(base,1)
        block_idx_big   = tf.where(mask_big,  f_range // tf.maximum(base + 1, 1), tf.zeros_like(f_range))
        block_idx_small = tf.where(mask_small,(f_range - cut) // tf.maximum(base, 1), tf.zeros_like(f_range))

        #Se mapea (t,f) -> UE usando order_mat
        #ue_big = order_mat[t, block_idx_big[f]]
        ue_big   = tf.gather(order_mat, block_idx_big, axis=1)         #[T, F]

        #ue_small = order_mat[t, rem + block_idx_small[f]]
        ue_small = tf.gather(order_mat, rem + block_idx_small, axis=1) #[T, F]

        #Selección por tramo con broadcasting de la máscara [F] -> [1,F] sobre T
        ue_idx_t_f = tf.where(tf.expand_dims(mask_big, 0), ue_big, ue_small)  #[T, F]

        #Conversión a codificación One-Hot para generar la máscara booleana de UEs
        onehot_ue_num = tf.one_hot(ue_idx_t_f, depth=U, dtype=tf.int32)    #[T, F, U], int32
        onehot_ue = tf.cast(onehot_ue_num, tf.bool)                        #[T, F, U], bool

        #Expansión dimensional para ajustarse al layout de Sionna SYS
        onehot_ue = tf.expand_dims(onehot_ue, axis=0)                  #[1, T, F, U]
        onehot_ue = tf.tile(onehot_ue, [num_bs, 1, 1, 1])     #[num_bs, T, F, U]

        #Aislamiento del Flujo Espacial Primario (Stream 0)
        num_streams = tf.cast(self.num_ut_ant, tf.int32)
        stream0 = tf.one_hot(0, depth=num_streams, dtype=tf.int32)         #[S] numérico
        stream0 = tf.cast(stream0, tf.bool)                                #[S] bool
        stream0 = tf.reshape(stream0, [1, 1, 1, 1, num_streams])    #[1,1,1,1,S]
        stream0 = tf.tile(stream0, [num_bs, T, F, U, 1])          #[num_bs,T,F,U,S]

        #Producto lógico: (Asignación en Frecuencia/Tiempo) AND (Stream 0)
        allocation_mask = tf.expand_dims(onehot_ue, axis=-1) & stream0     #[num_bs, T, F, U, S] bool
        return allocation_mask

    def set_velocities(
        self,
        doppler_enabled: bool,
        drone_velocity_mps: tuple[float, float, float],
        rx_velocities_mps: list[tuple[float, float, float]],
    ) -> None:
        """
        Inyecta los vectores de velocidad instantánea en los transmisores y receptores (objetos)
        del motor RT.

        Este metodo no altera las coordenadas espaciales, únicamente actualiza el
        estado requerido por el PathSolver para calcular los desplazamientos de
        fase continuos (Efecto Doppler) entre símbolos OFDM.
        """
        if self.scene is None:
            return

        try:
            vtx = (0.0, 0.0, 0.0) if not doppler_enabled else drone_velocity_mps
            if hasattr(self, "txs") and self.txs:
                #Lista de TX. Aplica misma v a todos los sectores del dron
                for tx in self.txs:
                    tx_velocities = [float(vtx[0]), float(vtx[1]), float(vtx[2])]
            elif hasattr(self.scene, "tx") and self.scene.tx is not None:
                self.scene.tx_velocities = [float(vtx[0]), float(vtx[1]), float(vtx[2])]
        except Exception as e:
            print("[WARN] set_velocities: no se pudo setear TX.velocity:", e)

        #Lista de RX
        try:
            if hasattr(self, "rx_list") and self.rx_list:
                N = min(len(self.rx_list), len(rx_velocities_mps))
                for i in range(N):
                    vrx = (0.0, 0.0, 0.0) if not doppler_enabled else rx_velocities_mps[i]
                    self.rx_list[i].velocity = [float(vrx[0]), float(vrx[1]), float(vrx[2])]
        except Exception as e:
            print("[WARN] set_velocities: no se pudo setear RX[i].velocity:", e)

    # ============================================================
    # 🔹 Métricas adicionales: Cálculo teórico de Potencia
    # ============================================================
    def compute_tx_rx_distances(self) -> np.ndarray:
        """
        Calcula la distancia Euclideana directa entre el Dron (TX) y todos los UEs (RX).

        Returns:
            np.ndarray: Vector 1D con las distancias en metros.
        """
        assert self.txs and self.rx_list, "Faltan TX y/o RX. Llama a build_scene() y attach_receivers()."
        txp = np.array(self.txs[0].position, dtype=float)
        rxp = np.array([list(rx.position) for rx in self.rx_list], dtype=float)
        d = np.linalg.norm(rxp - txp, axis=1)
        return d

    def compute_prx_dbm_theoretical(self,
                                    gamma: float = None,
                                    d0: float = 1.0,
                                    Gt_dBi: float = None,
                                    Gr_dBi: float = None) -> np.ndarray:
        """
        Calcula la Potencia Recibida (PRx) teórica usando el modelo Log-Distance Pathloss.
        Sirve como línea base (Baseline) teórica (basada en el modelo de Goldsmith) 
        para contrastar contra los resultados empíricos obtenidos vía Ray Tracing.

        Ecuación:
            PRx[dBm] = Pt[dBm] + K[dB] - 10*γ*log10(d/d0)
        donde K[dB] = 20*log10(λ/(4π d0)) + Gt + Gr
        """
        # Parámetros por defecto desde el sistema si no se proveen
        if gamma is None:
            gamma = getattr(self, "pathloss_gamma", 3.0) #Exponente de pérdida en entornos urbanos
        if Gt_dBi is None:
            Gt_dBi = float(getattr(self, "tx_gain_dbi", 0.0))
        if Gr_dBi is None:
            Gr_dBi = float(getattr(self, "rx_gain_dbi", 0.0))

        c = 299_792_458.0
        lam = c / float(self.freq_hz)

        #Cálculo de la constante K (Atenuación en espacio libre a la distancia de referencia d0)
        K_dB = 20.0 * math.log10(lam / (4.0 * math.pi * d0)) + Gt_dBi + Gr_dBi
        Pt_dBm = float(self._total_tx_power_dbm())

        #Posiciones actuales
        tx = np.array(self.txs[0].position, dtype=float).reshape(3)
        rx = np.array([list(r.position) for r in self.rx_list], dtype=float).reshape(-1, 3)
        d = np.linalg.norm(rx - tx, axis=1)

        ratio = np.maximum(d / float(d0), 1e-12)  #Evita singularidades matemáticas (log10(0))
        prx_dbm = Pt_dBm + K_dB - 10.0 * float(gamma) * np.log10(ratio)
        return np.asarray(prx_dbm, dtype=float).reshape(-1)

    # ============================================================
    # 🔹 Validación de movimiento sin colisión (dron)
    # ============================================================
    @staticmethod
    def _np3(p):
        """Helper para asegurar que el vector ingresado tenga 3 componentes [x,y,z]."""
        import numpy as np
        a = np.asarray(p, dtype=float).reshape(-1)
        if a.size != 3:
            raise ValueError("Se esperaban 3 componentes [x,y,z].")
        return a

    def is_move_valid(
        self,
        a, b,
        radius: float = 0.30,   #Radio estimado del chasis del dron (m)
        n_offsets: int = 12,    #Muestreo lateral alrededor del eje (0 = solo línea central)
        eps: float = 1e-3,      #;argen numérico para evitar autointersección de rayos
        check_bounds: bool = True
    ) -> bool:
        """
        Verifica la viabilidad física del movimiento A->B para el Dron.
        Utiliza el motor Mitsuba para disparar múltiples rayos alrededor del vector de
        desplazamiento, creando un "cilindro de colisión" virtual. Esto evita que el
        dron penetre mallas 3D (edificios, árboles) por sus bordes laterales.

        Returns:
            bool: True si el camino está libre, False en caso de colisión inminente
                  o salida de los límites (OutOfBounds).
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
            return True  #Sin desplazamiento efectivo

        #Chequeo de límites
        check_bounds = False
        if check_bounds:
            if getattr(self, "scene_bounds", None) is not None:
                pmin, pmax = self.scene_bounds
            else:
                pmin, pmax = self.scene_bounds_xyz()
            pmin = np.asarray(pmin, dtype=float); pmax = np.asarray(pmax, dtype=float)

            #Penalización por abandono de la zona de servicio permitida
            if np.any(b < (pmin - 1e-6)) or np.any(b > (pmax + 1e-6)):
                return False

        #Convertir coordenadas nativas a objetos vectoriales de Mitsuba
        a_mi = mi.Point3f(float(a[0]), float(a[1]), float(a[2]))
        b_mi = mi.Point3f(float(b[0]), float(b[1]), float(b[2]))
        dirv  = b_mi - a_mi
        L_mi  = dr.norm(dirv)
        dirv  = dirv / L_mi

        #Construcción de una base ortonormal perpendicular a la dirección de vuelo
        up = mi.Vector3f(0.0, 0.0, 1.0)
        n1 = dr.normalize(dr.cross(dirv, up))

        # Corrección de singularidad si el vuelo es puramente vertical (Eje Z)
        n1 = dr.select(dr.norm(n1) < 1e-6, dr.normalize(dr.cross(dirv, mi.Vector3f(0, 1, 0))), n1)
        n2 = dr.normalize(dr.cross(dirv, n1))

        # Cálculo de offsets radiales para aproximar el chasis físico o radio del Dron
        offsets = [mi.Vector3f(0.0, 0.0, 0.0)] #Rayo central
        if radius > 0.0 and n_offsets > 0:
            for k in range(int(n_offsets)):
                th = 2.0 * np.pi * (k / n_offsets)
                offsets.append(radius * np.cos(th) * n1 + radius * np.sin(th) * n2)

        mi_scene = self.scene.mi_scene
        L_lim = float(L) - eps

        #Disparo masivo de rayos (Ray-Casting)
        #Si cualquier rayo del cilindro intersecta un objeto, el movimiento se rechaza.
        for off in offsets:
            o = a_mi + off + eps * dirv
            ray = mi.Ray3f(o, dirv)
            ray.maxt = L_lim
            if mi_scene.ray_test(ray):
                return False
        #Ningún rayo intersecta
        return True
    
    # ============================================================
    # 🔹 Validación de movimiento sin colisión (receptores)
    # ============================================================
    @staticmethod
    def _np3_receptores(p):
        """Helper para asegurar vector [x,y,z] en np.float32 para los UEs."""
        import numpy as np
        a = np.asarray(p, dtype=np.float32).reshape(-1)
        if a.size != 3:
            raise ValueError("Se esperaban 3 componentes [x, y, z].")
        return a

    def is_move_valid_receptores(
            self,
            a, b,
            radius: float = 0.30,      #Radio físico promedio de un peatón
            n_offsets: int = 12,
            eps: float = 1e-3,
            check_bounds: bool = True
    ) -> bool:
        """
        Verifica la viabilidad física del movimiento A->B para los peatones (UEs).
        Utiliza el mismo principio de "cilindro de colisión" que el Dron, pero adaptado
        a las coordenadas y restricciones de los agentes terrestres gobernados por el SFM.
        """
        import numpy as np
        import mitsuba as mi
        import drjit as dr

        if self.scene is None:
            raise RuntimeError("SionnaRT: scene no está construida. Llama build_scene() antes.")

        a = self._np3_receptores(a)
        b = self._np3_receptores(b)
        d = b - a
        L = float(np.linalg.norm(d))
        if L <= 1e-9:
            return True  #Sin desplazamiento efectivo

        #Chequeo de límites
        if check_bounds:
            if getattr(self, "scene_bounds", None) is not None:
                pmin, pmax = self.scene_bounds
            else:
                pmin, pmax = self.scene_bounds_xyz()
            pmin = np.asarray(pmin, dtype=np.float32)
            pmax = np.asarray(pmax, dtype=np.float32)

            #Penalización por abandono de la zona de servicio permitida
            if np.any(b < (pmin - 1e-6)) or np.any(b > (pmax + 1e-6)):
                return False

        #Convertir coordenadas nativas a objetos vectoriales de Mitsuba
        a_mi = mi.Point3f(float(a[0]), float(a[1]), float(a[2]))
        b_mi = mi.Point3f(float(b[0]), float(b[1]), float(b[2]))
        dirv = b_mi - a_mi
        L_mi = dr.norm(dirv)
        dirv = dirv / L_mi

        # Construcción de una base ortonormal perpendicular a la dirección del movimiento
        up = mi.Vector3f(0.0, 0.0, 1.0)
        n1 = dr.normalize(dr.cross(dirv, up))
        n1 = dr.select(dr.norm(n1) < 1e-6,
                       dr.normalize(dr.cross(dirv, mi.Vector3f(0, 1, 0))),
                       n1)
        n2 = dr.normalize(dr.cross(dirv, n1))

        # --- Offsets circulares ---
        offsets = [mi.Vector3f(0.0, 0.0, 0.0)]
        if radius > 0.0 and n_offsets > 0:
            for k in range(int(n_offsets)):
                th = 2.0 * np.pi * (k / n_offsets)
                offsets.append(radius * np.cos(th) * n1 + radius * np.sin(th) * n2)

        mi_scene = getattr(self, "mi_scene", getattr(self.scene, "mi_scene", None))
        if mi_scene is None:
            raise RuntimeError("No hay escena Mitsuba activa (mi_scene=None)")

        L_lim = float(L) - eps
        for off in offsets:
            o = a_mi + off + eps * dirv
            ray = mi.Ray3f(o, dirv)
            ray.maxt = L_lim
            if mi_scene.ray_test(ray):
                return False

        return True

    # ============================================================
    # Escaner de la escena (Slicer / Ray-Casting)
    # ============================================================
    def get_sfm_obstacles(self, grid_density: float = 0.4) -> list[np.ndarray]:
        """
        Escanea la geometría de la escena 3D utilizando Ray Tracing vertical para generar
        un mapa de ocupación 2D preciso para la navegación peatonal (mapa de puntos 2D).

        Funcionamiento:
        1. Genera una cuadrícula de puntos sobre toda la escena.
        2. Lanza rayos desde arriba hacia abajo (eje -Z).
        3. Filtra los impactos según la altura para distinguir 'suelo caminable' de 'obstáculos'.

        Args:
            grid_density (float): Resolución del escaneo en metros.

        Returns:
            list[np.ndarray]: Lista conteniendo un array (N, 2) con las coordenadas X,Y
                              de todos los puntos detectados como obstáculos.
        """
        import numpy as np
        import mitsuba as mi
        import drjit as dr

        #Se valida si es que la escena esta cargada
        if self.scene is None or self.mi_scene is None:
            raise RuntimeError("SionnaRT: La escena no está construida.")

        print(f"[SionnaRT Slicer] Iniciando escaneo de escena (Densidad: {grid_density}m)")

        #1.-Se define el área de escaneo
        #Se obtienen los limites de toda la escena 3D de Sionna
        bounds = self.mi_scene.bbox()

        #Se añade un margen de +/- 2m con tal de asegurar cobertura total de la escena
        min_x, min_y = bounds.min.x - 2.0, bounds.min.y - 2.0
        max_x, max_y = bounds.max.x + 2.0, bounds.max.y + 2.0

        #Se genera la cuadrícula de coordenadas con los puntos (X, Y) (Malla de puntos)
        X = np.arange(min_x, max_x, grid_density, dtype=np.float32)
        Y = np.arange(min_y, max_y, grid_density, dtype=np.float32)
        grid_x, grid_y = np.meshgrid(X, Y)

        #Se aplanan las matrices para tener listas lineales de coordenadas
        flat_x = grid_x.flatten()
        flat_y = grid_y.flatten()

        #2.-Se configuran los rayos (Mitsuba / DrJit)
        #Altura de origen: Se coloca el "agente" 5 metros por encima del objeto más alto
        ray_origin_z = bounds.max.z + 5.0

        #Conversión - mi.Float convierte el array de NumPy al formato nativo de Mitsuba (Float)
        ox = mi.Float(flat_x)
        oy = mi.Float(flat_y)

        #'oz' es un escalar, pero DrJit realiza 'Broadcasting' automático.
        #Con la finalidad de expandir este valor único para coincidan con la longitud de ox y oy
        oz = mi.Float(ray_origin_z)

        #Origen de los rayos: (x, y, z) (z = Alto)
        origins_mi = mi.Point3f(ox, oy, oz)

        #Dirección de los rayos: Hacia abajo (0, 0, -1)
        dirs_mi = mi.Vector3f(0, 0, -1)

        #Se crea el objeto Rayo vectorizado (contiene miles de rayos)
        #Es un solo objeto que encapsula miles de rayos para cómputo paralelo.
        rays = mi.Ray3f(origins_mi, dirs_mi)

        #3. Intersección Masiva (Ray Tracing)
        #Mitsuba procesará todos los rayos en paralelo
        si = self.mi_scene.ray_intersect(rays)

        #4. Filtrado de obstáculos (Slicer)
        hit_z = si.p.z            #Coordenada Z donde golpeó el rayo (Altura del impacto).
        is_valid = si.is_valid()  #Booleano: ¿Golpeó algo o se fue al vacío?

        #Criterio de Obstáculos:
        #hit_z > 0.3: Se ignora el suelo (z = 0) y aceras o veredas muy bajas (< 30cm).
        #is_valid: Indica que golpeo algo (geometría).
        #No se limita la altura máxima para detectar correctamente techos de edificios altos (u otros obstáculos altos).
        is_obstacle = (hit_z > 0.3) & is_valid

        #Transferencia de datos: De DrJit a NumPy
        #Se convierte la máscara de DrJit a un array de NumPy
        obstacle_mask_np = np.array(is_obstacle, dtype=bool)

        #Se aplica la máscara para seleccionar solo las coordenadas X,Y que corresponden a obstáculos
        obs_x = flat_x[obstacle_mask_np]
        obs_y = flat_y[obstacle_mask_np]

        #Se apilan en formato (N, 2) para API Socialforce
        sfm_points = np.stack([obs_x, obs_y], axis=1)

        print(f"[SionnaRT Slicer] Escaneo completado: {len(sfm_points)} puntos de obstáculo detectados.")

        #Como la API Socialforce espera una lista de arrays (PedSpacePotential).
        #Se le devuelve una lista con un solo gran array de puntos.
        if len(sfm_points) > 0:
            return [sfm_points]
        else:
            return []