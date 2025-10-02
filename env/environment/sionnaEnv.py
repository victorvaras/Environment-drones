from __future__ import annotations
import math
import numpy as np
from pathlib import Path

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


# === Ruta A: utilidades PRB / ruido / SINR / goodput ===
#Funciones auxiliares para calculo de goodput

def _lin(db: float) -> float:
    return 10.0 ** (db / 10.0)

def _db(x: float) -> float:
    return 10.0 * math.log10(max(x, 1e-30))

def _prb_bandwidth_hz(scs_khz: float) -> float:
    """1 PRB = 12 subportadoras; BW = 12 * SCS."""
    return 12.0 * scs_khz * 1e3

def _noise_power_dbm_in_bw(bw_hz: float, noise_figure_db: float) -> float:
    """Potencia de ruido (dBm) en un ancho de banda 'bw_hz'."""
    return -174.0 + 10.0 * math.log10(bw_hz) + float(noise_figure_db)

def _goodput_routeA_bits(
    sinr_db: float,
    n_prb: int,
    scs_khz: float,
    dt_seconds: float,
    zeta: float = 0.85,
    overhead: float = 0.20,
    bler_target: float = 0.10,
) -> float:
    """Bits correctos en la ventana dt_seconds (Ruta A simple)."""
    eta = zeta * math.log2(1.0 + _lin(sinr_db)) * (1.0 - overhead)   # bit/s/Hz
    B_alloc = n_prb * _prb_bandwidth_hz(scs_khz)                      # Hz
    bits_tx = eta * B_alloc * dt_seconds
    return max((1.0 - bler_target) * bits_tx, 0.0)




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
                 tx_power_dbm: float = 30.0,      # Potencia TOTAL objetivo [dBm] (se reparte si hay sectores)
                 noise_figure_db: float = 7.0,    # NF del receptor [dB]
                 bandwidth_hz: float = 20e6,      # Ancho de banda de ruido térmico [Hz]

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
                num_subcarriers: int = 256,     # número de subportadoras
                num_ofdm_symbols: int = 24,     # número de símbolos OFDM
                bler_target: float = 0.1,       # objetivo de BLER para el enlace
                mcs_table_index: int = 1,       # índice de la tabla MCS a utilizar
                num_ut_ant: int = 1,            # número de antenas por usuario
                num_bs: int = 1,                # número de estaciones base                
                subcarrier_spacing: float = 30e3,
                **kwargs
               ):

        # --- Modo ---
        self.antenna_mode = str(antenna_mode).upper()

        # --- RF / ruido ---
        self.freq_hz = frequency_mhz * 1e6
        self.tx_power_dbm_total = tx_power_dbm     # guardamos la potencia total
        self.noise_figure_db = noise_figure_db
        self.bandwidth_hz = bandwidth_hz

        # --- escena / antenas ---
        self.scene_name = scene_name

        self.tx_array_rows = tx_array_rows
        self.tx_array_cols = tx_array_cols
        self.tx_array_v_spacing = tx_array_v_spacing
        self.tx_array_h_spacing = tx_array_h_spacing
        self.tx_array_pattern = tx_array_pattern
        self.tx_array_polarization = tx_array_polarization

        self.rx_array_rows = rx_array_rows
        self.rx_array_cols = rx_array_cols
        self.rx_array_v_spacing = rx_array_v_spacing
        self.rx_array_h_spacing = rx_array_h_spacing
        self.rx_array_pattern = rx_array_pattern
        self.rx_array_polarization = rx_array_polarization

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
            beta=0.01
        )

        self.num_ofdm_symbols = num_ofdm_symbols
        self.num_subcarriers = num_subcarriers
        self.subcarrier_spacing = subcarrier_spacing

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
        self.subcarrier_spacing = subcarrier_spacing
        self.num_subcarriers = num_subcarriers  # cantidad de subcanales OFDM (ejemplo 128)
        self.num_ofdm_symbols = num_ofdm_symbols  # cantidad de símbolos OFDM (ejemplo 12)

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
        """Crea 1 TX (ISO) o 3 TX sectoriales (SECTOR3_3GPP) y los añade a la escena."""
        self.txs = []

        if self.antenna_mode == "ISO":
            tx = Transmitter(name="tx0",
                             position=list(self.tx_initial_position),
                             display_radius=2)
            tx.orientation = list(self.tx_orientation_deg)  # [yaw, pitch, roll] en grados
            tx.power_dbm = float(self.tx_power_dbm_total)   # toda la potencia
            self.scene.add(tx)
            self.txs.append(tx)
            self.tx = tx  # compat

        elif self.antenna_mode in ("SECTOR3_3GPP", "SECTOR3", "3GPP"):
            # Reparto de potencia total en 3 sectores
            p_per_dbm = float(self.tx_power_dbm_total) - 10.0 * math.log10(3.0)
            base_yaw, base_pitch, base_roll = self.tx_orientation_deg
            sector_yaws = (0.0, 120.0, 240.0)  # 3 sectores a 120°
            for i, yaw_add in enumerate(sector_yaws):
                tx = Transmitter(name=f"tx{i}",
                                 position=list(self.tx_initial_position),
                                 display_radius=2)
                tx.orientation = [base_yaw + yaw_add, base_pitch, base_roll]
                tx.power_dbm = p_per_dbm
                self.scene.add(tx)
                self.txs.append(tx)
            self.tx = self.txs[0]  # compat

        else:
            raise ValueError(f"antenna_mode inválido: {self.antenna_mode}")

    def attach_receivers(self, rx_positions_xyz: np.ndarray):
        assert self.scene is not None, "build_scene() no fue llamado."
        self.rx_list = []
        for i, p in enumerate(rx_positions_xyz):
            rx = Receiver(name=f"RX_{i}",
                          position=[float(p[0]), float(p[1]), float(p[2])],
                          display_radius=1.5)
            self.scene.add(rx)
            self.rx_list.append(rx)

    def move_tx(self, pos_xyz):
        """Mueve TODOS los TX (1 o 3) a la misma posición del dron."""
        assert self.txs, "TX no inicializados."
        pos = [float(pos_xyz[0]), float(pos_xyz[1]), float(pos_xyz[2])]
        for tx in self.txs:
            tx.position = pos
        # Mantiene orientaciones según el modo (no se recalculan aquí)

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

    def compute_snr_db(self, prx_dbm: np.ndarray) -> np.ndarray:
        noise_dbm = -174.0 + 10.0 * math.log10(self.bandwidth_hz) + self.noise_figure_db
        return prx_dbm - noise_dbm

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
        # Si la escena lo expone, usa su AABB
        try:
            aabb = self.scene.aabb
            mn, mx = aabb[0], aabb[1]
            cx = float((mn[0] + mx[0]) / 2)
            cy = float((mn[1] + mx[1]) / 2)
            size_xy = max(float(mx[0] - mn[0]), float(mx[1] - mn[1]))
            z = max(150.0, size_xy * z_scale)
            return Camera(position=[cx, cy, z], look_at=[cx, cy, 0.0])
        except Exception:
            # Fallback fijo
            return Camera(position=[0, 0, 300], look_at=[0, 0, 0])


    # ---- Métricas adicionales Calculo teorico de Pr----

    # --- Distancia TX->cada RX (en metros) ---
    def compute_tx_rx_distances(self) -> np.ndarray:
        assert self.txs and self.rx_list, "Faltan TX y/o RX. Llama a build_scene() y attach_receivers()."
        txp = np.array(self.txs[0].position, dtype=float)
        rxp = np.array([list(rx.position) for rx in self.rx_list], dtype=float)
        d = np.linalg.norm(rxp - txp, axis=1)
        return d

    # --- PRx teórico (Goldsmith 2.40): Pt_dBm + K_dB - 10*gamma*log10(d/d0) ---
    def compute_prx_dbm_theoretical(self,
                                    gamma: float = 2.0,
                                    d0: float = 1.0,
                                    Gt_dBi: float = 0.0,
                                    Gr_dBi: float = 0.0) -> np.ndarray:
        
        c = 299_792_458.0
        lam = c / float(self.freq_hz)  # λ = c/f

        # K_dB = Friis @ d0 + ganancias
        K_dB = 20.0 * np.log10(lam / (4.0 * math.pi * d0)) + Gt_dBi + Gr_dBi

        Pt_dBm = self._total_tx_power_dbm()

        # Distancias TX->RX
        tx = np.array(self.txs[0].position, dtype=float)
        rx = np.array([list(r.position) for r in self.rx_list], dtype=float)
        d = np.linalg.norm(rx - tx, axis=1)

        ratio = np.maximum(d / d0, 1e-12)  # evita log10(0)
        prx_dbm = Pt_dBm + K_dB - 10.0 * float(gamma) * np.log10(ratio)


        return np.asarray(prx_dbm, dtype=float).reshape(-1)
    

    def estimate_ue_sinr_db_routeA(
        self,
        ue_index: int,
        scs_khz: float = 30.0,
        interference_dbm: float | None = None,
        ) -> float:
        """
        SINR efectivo por PRB para 'ue_index' usando Ruta A (simple).
        NOTA: tu compute_prx_dbm() devuelve PRx TOTAL en la banda 'self.bandwidth_hz'.
        Para un PRB, asumimos espectro plano y escalamos: S_PRB = S_total * (B_PRB / B_total).
        """
        prx_total_dbm = float(self.compute_prx_dbm()[ue_index])

        # Escala de potencia a 1 PRB asumiendo PSD plana:
        B_total = float(self.bandwidth_hz)
        B_prb = _prb_bandwidth_hz(scs_khz)
        # Potencia por PRB en dBm: P_total + 10*log10(B_prb/B_total)
        s_prb_dbm = prx_total_dbm + 10.0 * math.log10(max(B_prb / max(B_total, 1e-30), 1e-30))

        # Ruido térmico en 1 PRB + NF
        n_prb_dbm = _noise_power_dbm_in_bw(B_prb, self.noise_figure_db)

        if interference_dbm is None:
            # SINR dB ≈ S_prb(dBm) - N_prb(dBm)
            return s_prb_dbm - n_prb_dbm

        # Con interferencia: trabajar en lineal (mW)
        s_mw = _lin(s_prb_dbm)
        n_mw = _lin(n_prb_dbm)
        i_mw = _lin(interference_dbm)
        sinr_lin = s_mw / (n_mw + i_mw + 1e-30)
        return _db(sinr_lin)
    



    # --- Calculo de SYS
    @tf.function#(jit_compile=True)
    def sys_step(self, h, harq_feedback, sinr_eff_feedback, num_decoded_bits):
        """
        Ejecuta un step del sistema usando Sionna SYS.
        Calcula y devuelve métricas de bloques por step y acumuladas.
        """
        # --- Ruido (potencia por subportadora) ---
        no = BOLTZMANN_CONSTANT * 294 * self.bandwidth_hz

        # --- Ganancia de canal y tasa Shannon estimada (para scheduler PF) ---
        channel_gain = tf.math.square(tf.abs(h))  # |H|^2
        rate_achievable_est = log2(1.0 + channel_gain / no)
        # reduce antenas y tiempo-ofdm (coincide con tu versión)
        rate_achievable_est = tf.reduce_mean(rate_achievable_est, axis=[-3, -5])
        # -> [num_bs, num_ofdm_symbols, num_subcarriers, num_ut]
        rate_achievable_est = tf.transpose(rate_achievable_est, [1, 2, 3, 0])

        # --- Scheduler (PF) ---
        # is_scheduled: [num_bs, num_ofdm_symbols, num_subcarriers, num_ut, num_streams_per_ut]
        is_scheduled = self.scheduler(num_decoded_bits, rate_achievable_est)
        # ut por RE
        ut_scheduled = tf.argmax(tf.reduce_sum(tf.cast(is_scheduled, tf.int32), axis=-1), axis=-1)
        # REs asignados por BS y UT: [num_bs, num_ut]
        num_allocated_re = tf.reduce_sum(tf.cast(is_scheduled, tf.int32), axis=[-4, -3, -1])

        # --- Pathloss medio por UT (heurístico) ---
        pathloss_per_ut = tf.reduce_mean(1.0 / channel_gain, axis=[1, 3, 4, 5])  # [num_ut, num_bs]
        pathloss_per_ut = tf.transpose(pathloss_per_ut, [1, 0])                  # [num_bs, num_ut]

        # --- Control de potencia DL (fairness=0 => suma de throughput) ---
        tx_power_per_ut, _ = downlink_fair_power_control(
            pathloss_per_ut, no, num_allocated_re,
            bs_max_power_dbm=self.tx_power_dbm_total,
            guaranteed_power_ratio=0.5,
            fairness=0
        )

        # (Opcional) REs por usuario sumados en ejes OFDM (no usado en bloques)
        num_re_per_user = tf.reduce_sum(tf.cast(is_scheduled, tf.int32), axis=[0, 1, 2])

        # --- Reparto de potencia en REs asignados ---
        tx_power = spread_across_subcarriers(
            tf.expand_dims(tx_power_per_ut, axis=-2),  # [num_bs, num_ut, 1]
            is_scheduled,
            num_tx=self.num_bs
        )

        # --- Precoding + SINR post-ecualización ---
        precoded_channel = self.precoded_channel
        lmmse_posteq_sinr = self.lmmse_posteq_sinr

        h_eff = precoded_channel(h[tf.newaxis, ...], tx_power=tx_power, alpha=no)
        sinr = lmmse_posteq_sinr(h_eff, no=no, interference_whitening=True)
        # sinr: [num_bs, num_ofdm_symbols, num_subcarriers, num_ut, num_streams_per_ut]

        # --- Link Adaptation (OLLA) ---
        mcs_index = self.olla(
            num_allocated_re=num_allocated_re,
            sinr_eff=sinr_eff_feedback,
            mcs_table_index=self.mcs_table_index,
            mcs_category=1,  # downlink
            harq_feedback=harq_feedback
        )

        # --- Abstracción PHY: bits decodados, HARQ, SINR efectivo ---
        num_decoded_bits, harq_feedback, sinr_eff_true, *_ = self.phy_abs(
            mcs_index, sinr=sinr, mcs_table_index=self.mcs_table_index, mcs_category=1
        )

        sinr_eff_db_true = lin_to_db(sinr_eff_true)
        # feedback para OLLA en el próximo step (0 si no fue agendado)
        sinr_eff_feedback = tf.where(num_allocated_re > 0, sinr_eff_true, 0)

        # --- Eficiencias ---
        mod_order, coderate = decode_mcs_index(
            mcs_index, table_index=self.mcs_table_index, is_pusch=False
        )
        se_la = tf.where(harq_feedback == 1,
                        tf.cast(mod_order, coderate.dtype) * coderate,
                        tf.cast(0, tf.float32))
        se_shannon = log2(1 + sinr_eff_true)

        pf_metric = self.scheduler.pf_metric

        # --- BLER por usuario (1 - HARQ_OK) ---
        harq_01 = tf.clip_by_value(tf.cast(harq_feedback, tf.int32), 0, 1)  # [num_bs, num_ut]
        bler_per_user = 1.0 - tf.cast(harq_01, tf.float32)                  # [num_bs, num_ut]

        # --- Goodput “en bits” por usuario (bits ok) ---
        goodput_per_user = tf.cast(num_decoded_bits, tf.float32) * (1.0 - bler_per_user)

        # ============================================================
        # === NUEVO: Bloques por step y acumulados (por UE y total) ==
        # ============================================================
        # Intento (TX) si el UE tuvo algún RE asignado en el step
        # Shapes: num_allocated_re / harq_feedback / num_decoded_bits: [num_bs, num_ut]
        
        num_allocated_re_bs0 = num_allocated_re[0, :]                 # [num_ut], int32
        blocks_tx_step_per_ue = tf.cast(num_allocated_re_bs0 > 0, tf.int32)   # 0/1

        # Éxito: HARQ == 1 (clip explícito a {0,1}); si no hubo TX, debe quedar 0
        harq_ok_bs0 = tf.cast(tf.equal(harq_feedback[0, :], 1), tf.int32)      # 0/1 (NACK/ACK)
        blocks_ok_step_per_ue = blocks_tx_step_per_ue * harq_ok_bs0            # 0/1

        # % éxito por UE en el step
        success_rate_step_per_ue = tf.where(
            blocks_tx_step_per_ue > 0,
            tf.cast(blocks_ok_step_per_ue, tf.float32) / tf.cast(blocks_tx_step_per_ue, tf.float32),
            tf.zeros_like(tf.cast(blocks_ok_step_per_ue, tf.float32))
        )

        # Totales del step
        blocks_tx_step_total = tf.reduce_sum(blocks_tx_step_per_ue)  # escalar
        blocks_ok_step_total = tf.reduce_sum(blocks_ok_step_per_ue)  # escalar
        success_rate_step_total = tf.where(
            blocks_tx_step_total > 0,
            tf.cast(blocks_ok_step_total, tf.float32) / tf.cast(blocks_tx_step_total, tf.float32),
            tf.constant(0.0, dtype=tf.float32)
        )

        step_blocks = {
            "blocks_tx_step_per_ue": blocks_tx_step_per_ue,
            "blocks_ok_step_per_ue": blocks_ok_step_per_ue,
            "success_rate_step_per_ue": success_rate_step_per_ue,
            "blocks_tx_step_total": blocks_tx_step_total,
            "blocks_ok_step_total": blocks_ok_step_total,
            "success_rate_step_total": success_rate_step_total,
        }

        return (harq_feedback, sinr_eff_feedback, num_decoded_bits, mcs_index,
                se_la, se_shannon, sinr_eff_db_true, pf_metric, ut_scheduled,
                bler_per_user, goodput_per_user, num_re_per_user,
                step_blocks)  # <<--- NUEVO


    def get_current_channel_tensor(self):
        """
        Transforma el CFR guardado a tensor tf para SYS.
        """
        if not hasattr(self, 'current_cfr_for_current_step'):
            raise ValueError("No se ha actualizado current_cfr_for_current_step aún.")

        h_numpy = self.current_cfr_for_current_step
        h_tensor = tf.convert_to_tensor(h_numpy, dtype=tf.complex64)
        return h_tensor


    def run_sys_step(self):
        # Actualiza CFR para el step actual y obtén tensor de canal
        self.update_and_store_cfr_for_step()
        h = self.get_current_channel_tensor()

        (self.harq_feedback,
        self.sinr_eff_feedback,
        self.num_decoded_bits,
        mcs_index,
        se_la,
        se_shannon,
        sinr_eff_db_true,
        pf_metric,
        ut_scheduled,
        bler_per_user,
        goodput_per_user,
        num_re_per_user,
        step_blocks) = self.sys_step(
            h=h,
            harq_feedback=self.harq_feedback,
            sinr_eff_feedback=self.sinr_eff_feedback,
            num_decoded_bits=self.num_decoded_bits
        )

        # ---------- Inicializar acumuladores si no existen ----------
        if not hasattr(self, "blocks_acc_tx"):
            self.blocks_acc_tx = [0 for _ in range(self.num_ut)]  # intentos (TX) acumulados
        if not hasattr(self, "blocks_acc_ok"):
            self.blocks_acc_ok = [0 for _ in range(self.num_ut)]  # éxitos (OK) acumulados
        if not hasattr(self, "bits_acc_total"):
            self.bits_acc_total = [0 for _ in range(self.num_ut)] # bits ok acumulados

        # ---------- Pasar a numpy para armar métricas ----------
        sinr_eff_db_true_np = sinr_eff_db_true.numpy()[0, :]    # [num_ut]
        se_la_np            = se_la.numpy()[0, :]               # [num_ut]
        se_shannon_np       = se_shannon.numpy()[0, :]          # [num_ut]
        mcs_index_np        = mcs_index.numpy()[0, :]           # [num_ut]
        bler_np             = bler_per_user.numpy()[0, :]       # [num_ut]
        goodput_np          = goodput_per_user.numpy()[0, :]    # [num_ut]
        bits_np             = self.num_decoded_bits.numpy()[0, :]# [num_ut]

        blocks_tx_step_per_ue_np   = step_blocks["blocks_tx_step_per_ue"].numpy()   # [num_ut]
        blocks_ok_step_per_ue_np   = step_blocks["blocks_ok_step_per_ue"].numpy()   # [num_ut]
        success_rate_step_per_ue_np= step_blocks["success_rate_step_per_ue"].numpy()# [num_ut]

        blocks_tx_step_total_np    = int(step_blocks["blocks_tx_step_total"].numpy())
        blocks_ok_step_total_np    = int(step_blocks["blocks_ok_step_total"].numpy())
        success_rate_step_total_np = float(step_blocks["success_rate_step_total"].numpy())

        # ---------- Acumular ----------
        for i in range(self.num_ut):
            self.blocks_acc_tx[i] += int(blocks_tx_step_per_ue_np[i])
            self.blocks_acc_ok[i] += int(blocks_ok_step_per_ue_np[i])
            self.bits_acc_total[i] += int(bits_np[i])

        # % éxito acumulado por UE
        success_rate_accum_per_ue = [
            (self.blocks_acc_ok[i] / self.blocks_acc_tx[i]) if self.blocks_acc_tx[i] > 0 else 0.0
            for i in range(self.num_ut)
        ]

        # ---------- Métricas por-UE ----------
        ue_metrics = []
        prx_list = self.compute_prx_dbm()  # asumes que devuelve lista/array de len num_ut
        for i in range(self.num_ut):
            ue_metrics.append({
                "ue_id": i,
                "sinr_eff_db": float(sinr_eff_db_true_np[i]),
                "prx_dbm": float(prx_list[i]),
                "se_la": float(se_la_np[i]),
                "se_shannon": float(se_shannon_np[i]),
                "num_decoded_bits": int(bits_np[i]),
                "bler": float(bler_np[i]),
                "goodput": float(goodput_np[i]),
                "mcs_index": int(mcs_index_np[i]),

                # --- NUEVO: bloques por step y acumulados ---
                "blocks_tx_step": int(blocks_tx_step_per_ue_np[i]),
                "blocks_ok_step": int(blocks_ok_step_per_ue_np[i]),
                "success_rate_step": float(success_rate_step_per_ue_np[i]),

                "blocks_tx_accum": int(self.blocks_acc_tx[i]),
                "blocks_ok_accum": int(self.blocks_acc_ok[i]),
                "success_rate_accum": float(success_rate_accum_per_ue[i]),

                # Bits OK acumulados (útil para totals)
                "bits_ok_accum": int(self.bits_acc_total[i]),
            })

        # ---------- Resumen para render/tabla inferior derecha ----------
        step_blocks_summary = {
            "blocks_tx_step_total": blocks_tx_step_total_np,
            "blocks_ok_step_total": blocks_ok_step_total_np,
            "success_rate_step_total": success_rate_step_total_np,

            "blocks_tx_step_per_ue": [int(v) for v in blocks_tx_step_per_ue_np],
            "blocks_ok_step_per_ue": [int(v) for v in blocks_ok_step_per_ue_np],
            "success_rate_step_per_ue": [float(v) for v in success_rate_step_per_ue_np],

            "blocks_tx_accum_per_ue": [int(v) for v in self.blocks_acc_tx],
            "blocks_ok_accum_per_ue": [int(v) for v in self.blocks_acc_ok],
            "success_rate_accum_per_ue": [float(v) for v in success_rate_accum_per_ue],
            "bits_acc_total_per_ue": [int(v) for v in self.bits_acc_total],
        }

        return {
            "ue_metrics": ue_metrics,
            "pf_metric": pf_metric,
            "ut_scheduled": ut_scheduled,
            "step_blocks_summary": step_blocks_summary,  # << NUEVO
        }



    def update_and_store_cfr_for_step(self):
        """
        Calcula el canal CFR para el step actual y lo guarda para SYS.
        """
        # Obtención de paths
        paths = self._paths()
        # Definición de frecuencias para subports OFDM
        frequencies = subcarrier_frequencies(
            num_subcarriers=self.num_subcarriers,
            subcarrier_spacing=self.subcarrier_spacing
        )
        # Cálculo CFR en numpy
        h_freq = paths.cfr(
            frequencies=frequencies,
            sampling_frequency=1 / self.resource_grid.ofdm_symbol_duration,
            num_time_steps=self.num_ofdm_symbols,
            out_type="numpy"
        )  # Forma [num_rx, num_tx, num_ofdm_symbols, num_subcarriers]

        # Aquí puede ser necesario hacer reshape, expand_dims, transposición
        # para obtener una forma compatible con SYS
        self.current_cfr_for_current_step = h_freq


    
    