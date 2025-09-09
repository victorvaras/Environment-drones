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

# ---------- Utilidades ----------
def _resolve_scene_path(scene_id: str) -> str | None:
    """Devuelve ruta a XML si 'scene_id' es un archivo/carpeta válido
    o si existe dentro de scenes/. Si no encuentra, devuelve None."""
    p = Path(scene_id)
    if p.exists():
        if p.is_dir():
            xml = p / "scene.xml"
            if xml.exists():
                return str(xml)
        else:
            return str(p)
    base = Path(__file__).resolve().parents[1] / "scenes"
    cand_xml = base / f"{scene_id}.xml"
    if cand_xml.exists():
        return str(cand_xml)
    cand_dir = base / scene_id
    if cand_dir.is_dir():
        xml = cand_dir / "scene.xml"
        if xml.exists():
            return str(xml)
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
                 tx_initial_position: tuple[float, float, float] = (0.0, 0.0, 20.0), # [m]
                 tx_orientation_deg: tuple[float, float, float] = (0.0, -90.0, 0.0), # [°] yaw,pitch,roll

                 # --- control del trazador de caminos (PathSolver) ---
                 max_depth: int = 4,              # Nº máx. de interacciones por camino
                 los: bool = True,                # Considerar Line-of-Sight
                 specular_reflection: bool = True,# Reflexiones especulares (reflexiones tipo espejo)
                 diffuse_reflection: bool = False, # Reflexiones difusas, por superficies rugosas (muy costoso, realista)
                 refraction: bool = True,         # Refracción (atravesar vidrios, etc. cambiar angulo y atenuar)
                 synthetic_array: bool = False,   # True: matriz sintética (rápido); False: por elemento (en false realista)
                 samples_per_src: int | None = 500_000,    # Nº de rayos por fuente (default 1,000,000)
                 max_num_paths_per_src: int | None = None,  # Tope de caminos por fuente (None => default) (default 1000000)
                 seed: int = 41                   # Semilla del muestreo estocástico
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

    # ---- Construcción ----
    def build_scene(self):
        xml_path = _resolve_scene_path(self.scene_name)
        if xml_path is not None:
            scene = load_scene(xml_path, merge_shapes=True)
        else:
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
        cam = Camera(position=[0, 0, 300], look_at=[0, 0, 0])
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

