from pathlib import Path
import sys

# Ajuste de path a raíz del proyecto (dos niveles arriba)
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from env.environment.sionna import SionnaRT


if __name__ == "__main__":
    rt = SionnaRT(
        scene_name="simple_street_canyon_with_cars",
        antenna_mode="SECTOR3_3GPP", ) # modo de antena: "ISO", "SECTOR3_3GPP"
    rt.build_scene()
    
    #rt.render_scene_to_file(filename="simple_street_canyon_with_cars_200.png", resolution=(1000, 750), with_radio_map=False)
    
    # Mapa de calor de cobertura:
    rt.render_scene_to_file(filename="simple_street_canyon_with_cars_3-antenas.png", with_radio_map=True)
"""

if __name__ == "__main__":
    rt = SionnaRT(scene_name="munich")  # escena integrada "munich"
    rt.build_scene()
    rt.render_scene_to_file(filename="munich.png", resolution=(1000, 750), with_radio_map=False)
    # Si quieres mapa de calor de cobertura:
    rt.render_scene_to_file(filename="munich_coverage.png", with_radio_map=True)
"""