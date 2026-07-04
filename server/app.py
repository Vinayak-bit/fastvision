import torch
import timm
import time
import json
import io
import os
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from torchvision import transforms
from PIL import Image

# ── Load model on startup ──────────────────────────
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
    keep_indices = torch.tensor(keep_indices)
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
    keep_head_cols = torch.tensor(keep_head_cols)
    old_proj_weight = attn.proj.weight.data
    new_proj_weight = old_proj_weight[:, keep_head_cols].clone()
    new_proj = torch.nn.Linear(new_proj_weight.shape[1], embed_dim, bias=attn.proj.bias is not None)
    new_proj.weight.data = new_proj_weight
    if attn.proj.bias is not None:
        new_proj.bias.data = attn.proj.bias.data.clone()
    attn.proj = new_proj
    attn.num_heads = new_num_heads
    attn.attn_dim = new_num_heads * head_dim

def get_heads_to_prune(importance, ratio):
    all_heads = []
    for layer_idx, scores in importance.items():
        for head_idx, score in enumerate(scores):
            all_heads.append((score, layer_idx, head_idx))
    all_heads.sort(key=lambda x: x[0])
    n_prune = int(len(all_heads) * ratio)
    heads_to_prune = {}
    for _, layer_idx, head_idx in all_heads[:n_prune]:
        heads_to_prune.setdefault(layer_idx, []).append(head_idx)
    return heads_to_prune

def load_model():
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    # Load importance scores + rebuild pruned architecture
    importance_path = os.path.join(os.path.dirname(__file__), '..', 'results', 'head_importance.json')
    with open(importance_path) as f:
        importance = json.load(f)
    importance = {int(k): v for k, v in importance.items()}
    heads_to_prune = get_heads_to_prune(importance, ratio=0.25)

    model = timm.create_model('vit_base_patch16_224', pretrained=False)
    for layer_idx, heads in heads_to_prune.items():
        prune_attention_heads(model.blocks[layer_idx], set(heads))

    ckpt_path = os.path.join(os.path.dirname(__file__), '..', 'checkpoints', 'pruned_25_fp16.pt')
    state = torch.load(ckpt_path, map_location='cpu')
    # fp16 checkpoint -- convert state dict values to float16
    state = {k: v.half() if v.dtype == torch.float32 else v for k, v in state.items()}
    model.load_state_dict(state)
    model = model.half().to(device).eval()

    return model, device

print("Loading model...")
MODEL, DEVICE = load_model()
print(f"Model ready on {DEVICE}")

# ── Preprocessing ──────────────────────────────────
TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ── FastAPI app ────────────────────────────────────
app = FastAPI(title="FastVision", version="1.0")

@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": str(DEVICE),
        "model": "vit_base_patch16_224 pruned_25_fp16"
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    t_start = time.perf_counter()

    # Read + decode image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert('RGB')
    tensor = TRANSFORM(image).unsqueeze(0).half().to(DEVICE)

    # Inference
    t_infer = time.perf_counter()
    with torch.no_grad():
        logits = MODEL(tensor)
        if DEVICE.type == 'mps':
            torch.mps.synchronize()
    t_infer_end = time.perf_counter()

    probs = torch.softmax(logits.float(), dim=1)
    conf, pred = probs.max(1)

    total_ms = (time.perf_counter() - t_start) * 1000
    infer_ms = (t_infer_end - t_infer) * 1000

    return JSONResponse({
        "class_id":    pred.item(),
        "confidence":  round(conf.item(), 4),
        "infer_ms":    round(infer_ms, 2),   # pure model time
        "total_ms":    round(total_ms, 2),   # includes image decode + transfer
    })

@app.get("/benchmark")
def benchmark_endpoint():
    """Quick single-request latency check via random tensor -- no image upload needed."""
    dummy = torch.randn(1, 3, 224, 224).half().to(DEVICE)
    # warmup
    for _ in range(5):
        with torch.no_grad():
            MODEL(dummy)
    if DEVICE.type == 'mps':
        torch.mps.synchronize()

    import numpy as np
    latencies = []
    for _ in range(50):
        t = time.perf_counter()
        with torch.no_grad():
            MODEL(dummy)
        if DEVICE.type == 'mps':
            torch.mps.synchronize()
        latencies.append((time.perf_counter() - t) * 1000)

    return {
        "n_runs": 50,
        "p50_ms": round(float(np.percentile(latencies, 50)), 2),
        "p95_ms": round(float(np.percentile(latencies, 95)), 2),
        "p99_ms": round(float(np.percentile(latencies, 99)), 2),
        "model":  "pruned_25_fp16_mps",
    }
