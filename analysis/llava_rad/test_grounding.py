"""Test sintético de las métricas de anclaje: heatmap dentro vs fuera del bbox."""
import sys
sys.path.insert(0, r"./")
import numpy as np
from src.modules.grounding import evaluate_localization

# bbox en el cuadrante superior-izquierdo de una imagen 1024
bbox = (150, 150, 300, 300)  # x,y,w,h

# Heatmap 14x14 con pico DENTRO del bbox (superior-izq)
hm_in = np.zeros((14, 14), np.float32); hm_in[3, 3] = 1.0
hm_in = np.clip(hm_in + 0.05, 0, None)  # algo de fondo

# Heatmap con pico FUERA (inferior-der)
hm_out = np.zeros((14, 14), np.float32); hm_out[11, 11] = 1.0
hm_out = np.clip(hm_out + 0.05, 0, None)

s_in = evaluate_localization(hm_in, bbox)
s_out = evaluate_localization(hm_out, bbox)

print("DENTRO :", f"hit={s_in.pointing_hit} energy={s_in.energy:.3f} "
      f"iobb={s_in.iobb:.3f} base={s_in.baseline:.3f} lift={s_in.lift:.2f}")
print("FUERA  :", f"hit={s_out.pointing_hit} energy={s_out.energy:.3f} "
      f"iobb={s_out.iobb:.3f} base={s_out.baseline:.3f} lift={s_out.lift:.2f}")

assert s_in.pointing_hit and not s_out.pointing_hit, "pointing game mal"
assert s_in.energy > s_out.energy, "energy mal"
assert s_in.lift > 1.0, "lift dentro debe ser >1"
print("\n✓ Métricas de anclaje correctas.")
