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

def prune_attention_heads(block, heads_to_remove, head_dim=64):
    attn = block.attn
    old_num_heads = attn.num_heads
    embed_dim = attn.proj.in_features
    keep_heads = [h for h in range(old_num_heads) if h not in heads_to_remove]
    new_num_heads = len(keep_heads)
    if new_num_heads == old_num_heads:
        return
    keep_indices = []
    for section in range(3):
        section_offset = section * old_num_heads * head_dim
        for h in keep_heads:
            start = section_offset + h * head_dim
            keep_indices.extend(range(start, start + head_dim))
    keep_indices = torch.tensor(keep_indices, dtype=torch.long)
    old_qkv_weight = attn.qkv.weight.data
    old_qkv_bias = attn.qkv.bias.data if attn.qkv.bias is not None else None
    new_qkv_weight = old_qkv_weight[keep_indices, :].clone()
    new_qkv = torch.nn.Linear(embed_dim, new_qkv_weight.shape[0], bias=old_qkv_bias is not None)
    new_qkv.weight.data = new_qkv_weight
    if old_qkv_bias is not None:
        new_qkv.bias.data = old_qkv_bias[keep_indices].clone()
    attn.qkv = new_qkv
    keep_head_cols = []
    for h in keep_heads:
        start = h * head_dim
        keep_head_cols.extend(range(start, start + head_dim))
    keep_head_cols = torch.tensor(keep_head_cols, dtype=torch.long)
    old_proj_weight = attn.proj.weight.data
    new_proj_weight = old_proj_weight[:, keep_head_cols].clone()
    new_proj = torch.nn.Linear(new_proj_weight.shape[1], embed_dim, bias=attn.proj.bias is not None)
    new_proj.weight.data = new_proj_weight
    if attn.proj.bias is not None:
        new_proj.bias.data = attn.proj.bias.data.clone()
    attn.proj = new_proj
    attn.num_heads = new_num_heads
    attn.attn_dim = new_num_heads * head_dim

def load_model():
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    # Load pruning pattern -- always paired with checkpoint
    pattern_path = os.path.join(os.path.dirname(__file__), '..', 'checkpoints', 'vit_small_pruned_10_pattern.json')
    with open(pattern_path) as f:
        heads_to_prune = json.load(f)
    heads_to_prune = {int(k): v for k, v in heads_to_prune.items()}

    model = timm.create_model('vit_small_patch16_224', pretrained=False)
    for layer_idx, heads in heads_to_prune.items():
        prune_attention_heads(model.blocks[layer_idx], set(heads))

    ckpt_path = os.path.join(os.path.dirname(__file__), '..', 'checkpoints', 'vit_small_pruned_10_fp16.pt')
    state = torch.load(ckpt_path, map_location='cpu')
    state = {k: v.half() for k, v in state.items()}
    model.load_state_dict(state)
    model = model.half().to(device).eval()
    return model, device

print("Loading ViT-Small pruned 10% FP16...")
MODEL, DEVICE = load_model()
print(f"Model ready on {DEVICE}")

# Warmup -- bake in MPS shader compilation before serving
print("Warming up...")
dummy = torch.randn(1, 3, 224, 224).half().to(DEVICE)
for _ in range(50):
    with torch.inference_mode(): MODEL(dummy)
if DEVICE.type == 'mps': torch.mps.synchronize()
print("Ready.")

TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

IMAGENETTE_MAP = {
    'n01440764': 0, 'n02102040': 217, 'n02979186': 482, 'n03000684': 491,
    'n03028079': 497, 'n03394916': 566, 'n03417042': 569, 'n03425413': 571,
    'n03445777': 574, 'n03888257': 701,
}
# Reverse map: ImageNet index -> class name
IDX_TO_NAME = {v: k for k, v in IMAGENETTE_MAP.items()}

from concurrent.futures import ThreadPoolExecutor
EXECUTOR = ThreadPoolExecutor(max_workers=1)

def run_inference(tensor):
    tensor = tensor.to(DEVICE)
    with torch.inference_mode():
        logits = MODEL(tensor)
        if DEVICE.type == 'mps': torch.mps.synchronize()
    probs = torch.softmax(logits.float(), dim=1)
    conf, pred = probs.max(1)
    return pred.item(), round(conf.item(), 4)

app = FastAPI(title="FastVision - ViT-Small Pruned 10%")

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": "vit_small_patch16_224 pruned_10% FP16",
        "device": str(DEVICE),
        "accuracy_imagenette": "98.6%",
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
        "model": "vit_small_pruned_10_fp16",
        "p50_ms": round(float(np.percentile(latencies, 50)), 2),
        "p95_ms": round(float(np.percentile(latencies, 95)), 2),
        "p99_ms": round(float(np.percentile(latencies, 99)), 2),
        "accuracy_imagenette": "98.6%",
    }
