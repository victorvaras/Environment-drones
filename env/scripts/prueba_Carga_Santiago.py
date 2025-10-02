from pathlib import Path
import sys

# Ajuste de path a raíz del proyecto (dos niveles arriba)
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from env.environment.sionnaEnv import SionnaRT


if __name__ == "__main__":
    rt = SionnaRT(
        scene_name="santiago.xml",
        antenna_mode="SECTOR3_3GPP",  # modo de antena: "ISO", "SECTOR3_3GPP"
        tx_initial_position=(0, 0.0, 150.0)
    )
    rt.build_scene()

    #rt.render_scene_to_file(filename="santiago_200.png", resolution=(1000, 750), with_radio_map=False)

    # Mapa de calor de cobertura:
    rt.render_scene_to_file(filename="santiago_100metros.png", with_radio_map=True)