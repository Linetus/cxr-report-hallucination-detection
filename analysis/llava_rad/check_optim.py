import py_compile
py_compile.compile(r"./src/training/lightning_module.py", doraise=True)
print("compila OK")
import sys
sys.path.insert(0, r"./")
from src.models.model import RadiologyReportModel
m = RadiologyReportModel(encoder_pretrained=False, decoder_pretrained=False)
ca = [n for n, _ in m.named_parameters() if ("cross_attn" in n or "visual_proj" in n)]
gates = [n for n in ca if n.endswith("cross_attn.gate")]
print("params via visual:", len(ca), "| gates:", len(gates),
      "| visual_proj:", sum("visual_proj" in n for n in ca))
