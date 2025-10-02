# Forzar ejecución en CPU y reducir logs de TensorFlow
import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # No usar GPU
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

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
"""
# Importar/validar Sionna (sin lógica específica de Colab/Jupyter)
try:
    import sionna.phy
    import sionna.rt
except ImportError as e:
    sys.stderr.write("Sionna no está instalado. Instálalo con: pip install sionna\n")
    raise

# Configurar TensorFlow para CPU
import tensorflow as tf
try:
    tf.config.set_visible_devices([], "GPU")  # Ocultar GPUs a TF
except Exception:
    pass  # Puede fallar si los dispositivos ya fueron inicializados

# Evitar warnings de TensorFlow
tf.get_logger().setLevel("ERROR")

import numpy as np

# Importes Sionna para simulación a nivel de enlace
from sionna.phy.channel import OFDMChannel, CIRDataset
from sionna.phy.nr import PUSCHConfig, PUSCHTransmitter, PUSCHReceiver
from sionna.phy.utils import ebnodb2no, PlotBER
from sionna.phy.ofdm import KBestDetector, LinearDetector
from sionna.phy.mimo import StreamManagement

# Componentes de Sionna RT
from sionna.rt import (
    load_scene, Camera, Transmitter, Receiver, PlanarArray,
    PathSolver, RadioMapSolver
)
"""

no_preview = True  # Mantener sin preview (no Jupyter)

if __name__ == "__main__":
    # Punto de entrada para ejecución en CLI (Ubuntu)
    print("Entorno configurado para ejecutar en CPU.")
    # TODO: añade aquí la lógica principal de tu script
