"""Verifica el fix: con gate=0 la cross-attn es identidad → GPT-2 predice palabras."""
import sys
sys.path.insert(0, r"./")
import torch
torch.set_num_threads(4)
from transformers import AutoTokenizer
from src.models.model import RadiologyReportModel

tok = AutoTokenizer.from_pretrained("gpt2"); tok.pad_token = tok.eos_token

# Modelo fresco: encoder random (irrelevante, gate=0 anula lo visual), GPT-2 preentrenado
model = RadiologyReportModel(encoder_pretrained=False, decoder_pretrained=True).eval()

# Comprobar que los gates arrancan en 0
gates = [b.cross_attn.gate.item() for b in model.decoder.gpt2.transformer.h]
print("Gates (deben ser 0):", set(round(g, 4) for g in gates))

report = ("The heart, pulmonary XXXX and mediastinum are within normal limits. "
          "There is no pleural effusion or pneumothorax.")
enc = tok(report, return_tensors="pt")
ids, am = enc.input_ids, enc.attention_mask
img = torch.randn(1, 3, 224, 224)
n = ids.shape[1]

with torch.no_grad():
    out = model(pixel_values=img, input_ids=ids, attention_mask=am)  # CON inyección visual
pred = out.logits[0].argmax(-1)[:n-1]
print("\n[REAL   ]:", repr(tok.decode(ids[0][:30])))
print("[PRED-TF]:", repr(tok.decode(pred[:30])))
print(f"\nTokens únicos predichos: {len(set(pred.tolist()))}  (>>1 y palabras ⇒ FIX OK)")
