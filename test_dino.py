# test_dino.py
import torch

print("Loading DINOv2... (first time downloads ~85MB)")
dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', pretrained=True)
dino.eval()
print("DINOv2 loaded successfully!")

# Quick test
dummy = torch.randn(1, 3, 224, 224)
with torch.no_grad():
    out = dino(dummy)
print(f"Feature vector shape: {out.shape}")  # should be (1, 384)