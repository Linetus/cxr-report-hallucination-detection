# Demo TFM — Fiabilidad de informes radiológicos IA

Dos apps Gradio desplegables (una por modelo). **No ejecutar a la vez** (no caben 2 modelos en 12 GB de VRAM).

Cada app muestra 3 paneles: **informe generado**, **DÓNDE mira** (localización) y **CUÁNTO fiarse**
(semáforo-gradiente calibrado sobre `min_prob` de contenido = incertidumbre de la decisión del modelo).
Los ejemplos incluyen el **veredicto real del radiólogo** para comparar.

## App 1 — LLaVA-Rad + mapa de calor
```
conda activate llavarad   # env con LLaVA-Rad
python demo/app.py        # -> http://127.0.0.1:7860
```
- "DÓNDE": **mapa de calor** de atención (anclaje lift≈3.0). Coherente con el estudio de anclaje de la memoria.
- Semáforo calibrado en Open-I (`semaforo_calib.json`): 🔴≤0.34 (30% alucinación) · 🟡 · 🟩 · 🟢>0.48 (0%).

## App 2 — MAIRA-2 + cajas de anclaje
```
conda activate maira2     # env con MAIRA-2 (transformers 4.46)
python demo/app_maira.py  # -> http://127.0.0.1:7862
```
- "DÓNDE": **cajas de grounding propias de MAIRA** (objetivas, comparables con las cajas reales).
- Semáforo calibrado en MAIRA (`maira_semaforo_calib.json`): 🔴≤0.28 (14%) · 🟡 · 🟩 · 🟢>0.54 (0%).
- Requiere licencia aceptada de `microsoft/maira-2` en HF + `huggingface-cli login`.

## Mensaje clave (en la leyenda de ambas)
El color mide la **duda del modelo** (validado: AUC 0.80–0.90 en 2 modelos × 2 dominios).
Verde = baja probabilidad de **alucinación**, pero NO garantiza ausencia de **omisiones**
(un fallo confiado puede caer en verde). La localización indica **dónde**, no cuánto fiarse
— evita el sesgo de automatización.

## Notas de entorno
El env `maira2` usa gradio 3.50.2 + pydantic 1.10.13 (gradio 4.x falla el auto-chequeo de localhost
en este equipo). No afecta a la inferencia del modelo.

## Example images

The example X-rays used by the demos are not included, because the dataset licences (PadChest in particular) do not allow redistribution. Put your own examples in `demo/examples/` and `demo/examples_maira/` after obtaining the datasets.
