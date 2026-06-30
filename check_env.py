import torch
import timm

print("PyTorch version:", torch.__version__)
print("MPS available:", torch.backends.mps.is_available())
print("MPS built:", torch.backends.mps.is_built())

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print("Using device:", device)

model = timm.create_model('vit_base_patch16_224', pretrained=True)
model.eval()
print("Model loaded:", sum(p.numel() for p in model.parameters()), "parameters")
