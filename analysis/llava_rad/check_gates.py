"""¿Se han abierto los gates de cross-attention? (evidencia de que la visión se usa)."""
import glob, torch

ckpts = sorted(glob.glob(r"./outputs/checkpoints/baseline_v3/stage2_epoch=*.ckpt"))
path = ckpts[-1]
print("Checkpoint:", path.split("\\")[-1])
sd = torch.load(path, map_location="cpu")["state_dict"]

gates = {k: v for k, v in sd.items() if k.endswith("cross_attn.gate")}
print(f"\n{'capa':>5} | {'gate':>9} | {'tanh(gate)':>10}")
print("-" * 32)
vals = []
for k in sorted(gates, key=lambda s: int(s.split(".h.")[1].split(".")[0])):
    layer = int(k.split(".h.")[1].split(".")[0])
    g = gates[k].item()
    vals.append(abs(torch.tanh(gates[k]).item()))
    print(f"{layer:>5} | {g:>9.4f} | {torch.tanh(gates[k]).item():>10.4f}")

import statistics
print("-" * 32)
print(f"|tanh(gate)| medio: {statistics.mean(vals):.4f}  máx: {max(vals):.4f}")
print("Todos empezaron en 0.0 → si ahora son !=0, la cross-attention SE ESTÁ USANDO.")
