"""Paso A: mergear LoRA de LLaVA-Rad en CPU (fp16) y guardar el modelo fusionado."""
import os
os.environ["HF_HOME"] = r"./hf_cache/"
import torch, time
from transformers import AutoConfig, AutoTokenizer
from llava.model import LlavaLlamaForCausalLM
from peft import PeftModel
from huggingface_hub import hf_hub_download

MP = "microsoft/llava-rad"; MB = "lmsys/vicuna-7b-v1.5"
OUT = r"./hf_cache/llavarad-merged"
t0 = time.time()

cfg = AutoConfig.from_pretrained(MP)
tok = AutoTokenizer.from_pretrained(MB, use_fast=False)
print("Cargando base fp16 en CPU (sin device_map)...", flush=True)
model = LlavaLlamaForCausalLM.from_pretrained(MB, low_cpu_mem_usage=True, config=cfg, torch_dtype=torch.float16)

# mm_projector (non-lora trainables)
nlt = torch.load(hf_hub_download(MP, "non_lora_trainables.bin"), map_location="cpu")
nlt = {(k[11:] if k.startswith("base_model.") else k): v for k, v in nlt.items()}
if any(k.startswith("model.model.") for k in nlt):
    nlt = {(k[6:] if k.startswith("model.") else k): v for k, v in nlt.items()}
model.load_state_dict(nlt, strict=False)
print("mm_projector cargado. Aplicando LoRA...", flush=True)

model = PeftModel.from_pretrained(model, MP)   # CPU, sin device_map → sin ZeroDivisionError
print("Mergeando LoRA (fp16)...", flush=True)
model = model.merge_and_unload()
print(f"Merge OK en {time.time()-t0:.0f}s. Guardando en {OUT}...", flush=True)
model.save_pretrained(OUT, safe_serialization=True)
tok.save_pretrained(OUT)
print(f"✓ Modelo fusionado guardado ({time.time()-t0:.0f}s total).")
