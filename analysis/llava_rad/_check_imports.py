import importlib
for m in ["numpy","scipy","sklearn"]:
    try:
        mod=importlib.import_module(m); print("OK", m, getattr(mod,"__version__","?"))
    except Exception as e:
        print("FAIL", m, type(e).__name__, e)
