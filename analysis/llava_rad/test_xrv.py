import torch, torchxrayvision as xrv
m = xrv.models.DenseNet(weights="densenet121-res224-all").eval()
x = torch.randn(2, 1, 224, 224) * 500  # dummy en rango tipo xrv
with torch.no_grad():
    feat_map = m.features(x)     # (B,1024,7,7)
    pooled = m.features2(x)      # (B,1024) vector embedding
print("WEIGHTS_OK targets=", len(m.pathologies))
print("FEATURE_MAP", tuple(feat_map.shape))
print("POOLED_EMB", tuple(pooled.shape))
print("TORCH", torch.__version__, "CUDA_BUILD", torch.version.cuda)
