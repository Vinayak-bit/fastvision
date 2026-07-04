import torch
import timm
import time
import json
import io
import os
import asyncio
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from torchvision import transforms
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

IMAGENETTE_MAP = {
    'n01440764': 0, 'n02102040': 217, 'n02979186': 482, 'n03000684': 491,
    'n03028079': 497, 'n03394916': 566, 'n03417042': 569, 'n03425413': 571,
    'n03445777': 574, 'n03888257': 701,
}
IDX_TO_NAME = {v: k for k, v in IMAGENETTE_MAP.items()}

def load_model():
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model = timm.create_model('vit_tiny_patch16_224', pretrained=False)
    state = torch.load(
        os.path.join(os.path.dirname(__file__), '..', 'checkpoints', 'vit_tiny_finetuned_best.pt'),
        map_location='cpu'
    )
    state = {k: v.half() for k, v in state.items()}
    model.load_state_dict(state)
    model = model.half().to(device).eval()
    return model, device

print("Loading ViT-Tiny finetuned FP16...")
MODEL, DEVICE = load_model()

print("Warming up (50 runs)...")
dummy = torch.randn(1, 3, 224, 224).half().to(DEVICE)
for _ in range(50):
    with torch.inference_mode(): MODEL(dummy)
if DEVICE.type == 'mps': torch.mps.synchronize()
print("Ready.")

TRANSFORM = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

EXECUTOR = ThreadPoolExecutor(max_workers=1)

def run_inference(tensor):
    tensor = tensor.to(DEVICE)
    with torch.inference_mode():
        logits = MODEL(tensor)
        if DEVICE.type == 'mps': torch.mps.synchronize()
    probs = torch.softmax(logits.float(), dim=1)
    conf, pred = probs.max(1)
    return pred.item(), round(conf.item(), 4)

app = FastAPI(title="FastVision - ViT-Tiny Finetuned")

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": "vit_tiny_patch16_224 finetuned FP16",
        "device": str(DEVICE),
        "accuracy_imagenette": "99.0%",
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    t_start = time.perf_counter()
    loop = asyncio.get_event_loop()
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert('RGB')
    tensor = TRANSFORM(image).unsqueeze(0).half()
    t_infer = time.perf_counter()
    pred, conf = await loop.run_in_executor(EXECUTOR, run_inference, tensor)
    infer_ms = (time.perf_counter() - t_infer) * 1000
    total_ms = (time.perf_counter() - t_start) * 1000
    return JSONResponse({
        "class_id":   pred,
        "class_name": IDX_TO_NAME.get(pred, "unknown"),
        "confidence": conf,
        "infer_ms":   round(infer_ms, 2),
        "total_ms":   round(total_ms, 2),
    })

@app.get("/benchmark")
async def benchmark():
    loop = asyncio.get_event_loop()
    def _bench():
        t = torch.randn(1, 3, 224, 224).half().to(DEVICE)
        for _ in range(20):
            with torch.inference_mode(): MODEL(t)
        torch.mps.synchronize()
        latencies = []
        for _ in range(200):
            start = time.perf_counter()
            with torch.inference_mode(): MODEL(t)
            torch.mps.synchronize()
            latencies.append((time.perf_counter() - start) * 1000)
        return latencies
    latencies = await loop.run_in_executor(EXECUTOR, _bench)
    return {
        "model":   "vit_tiny_finetuned_fp16",
        "p50_ms":  round(float(np.percentile(latencies, 50)), 2),
        "p95_ms":  round(float(np.percentile(latencies, 95)), 2),
        "p99_ms":  round(float(np.percentile(latencies, 99)), 2),
        "accuracy_imagenette": "99.0%",
    }
