import torch
import timm
import time
import json
import io
import os
import asyncio
import numpy as np
from fastapi import FastAPI, UploadFile, File, Request
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
    state = {k: v.half() if v.dtype == torch.float32 else v for k, v in state.items()}
    model.load_state_dict(state)
    model = model.half().to(device).eval()
    return model, device

print("Loading model...")
MODEL, DEVICE = load_model()
print(f"Model ready on {DEVICE}")

TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def preprocess(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    return TRANSFORM(image).unsqueeze(0).half()


# ── Dynamic Batcher ────────────────────────────────
class DynamicBatcher:
    def __init__(self, model, device, max_batch_size=8, max_wait_ms=20):
        self.model = model
        self.device = device
        self.max_batch = max_batch_size
        self.max_wait = max_wait_ms / 1000.0
        self.queue = None
        self._task = None

    def _ensure_started(self):
        if self.queue is None:
            self.queue = asyncio.Queue()
            self._task = asyncio.create_task(self._batch_loop())

    async def predict(self, tensor):
        self._ensure_started()
        loop = asyncio.get_event_loop()
        future = loop.create_future()
        await self.queue.put((tensor, future))
        return await future

    async def _batch_loop(self):
        while True:
            tensor0, future0 = await self.queue.get()
            tensors = [tensor0]
            futures = [future0]
            deadline = asyncio.get_event_loop().time() + self.max_wait
            while len(tensors) < self.max_batch:
                remaining = deadline - asyncio.get_event_loop().time()
                if remaining <= 0:
                    break
                try:
                    tensor_n, future_n = await asyncio.wait_for(
                        self.queue.get(), timeout=remaining
                    )
                    tensors.append(tensor_n)
                    futures.append(future_n)
                except asyncio.TimeoutError:
                    break

            batch = torch.cat(tensors, dim=0).to(self.device)
            t_infer = time.perf_counter()
            try:
                with torch.no_grad():
                    logits = self.model(batch)
                    if self.device.type == 'mps':
                        torch.mps.synchronize()
                infer_ms = (time.perf_counter() - t_infer) * 1000
                probs = torch.softmax(logits.float(), dim=1)
                confs, preds = probs.max(1)
                for i, future in enumerate(futures):
                    if not future.done():
                        future.set_result({
                            "class_id":   preds[i].item(),
                            "confidence": round(confs[i].item(), 4),
                            "infer_ms":   round(infer_ms, 2),
                            "batch_size": len(tensors),
                        })
            except Exception as e:
                for future in futures:
                    if not future.done():
                        future.set_exception(e)


BATCHER = DynamicBatcher(MODEL, DEVICE, max_batch_size=8, max_wait_ms=20)

app = FastAPI(title="FastVision Batched", version="3.0")

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "device": str(DEVICE),
        "model": "vit_base_patch16_224 pruned_25_fp16",
        "serving": "async + dynamic batching",
        "max_batch_size": BATCHER.max_batch,
        "max_wait_ms": BATCHER.max_wait * 1000,
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Image upload endpoint -- includes preprocessing overhead."""
    t_start = time.perf_counter()
    contents = await file.read()
    tensor = preprocess(contents)
    result = await BATCHER.predict(tensor)
    result["total_ms"] = round((time.perf_counter() - t_start) * 1000, 2)
    return JSONResponse(result)

@app.post("/predict_tensor")
async def predict_tensor(request: Request):
    """
    Raw tensor endpoint -- skips image IO entirely.
    Accepts raw float16 bytes: shape [1, 3, 224, 224].
    Used for load testing pure inference + batching performance
    without image decode overhead contaminating the numbers.
    Client sends: tensor.numpy().tobytes()
    """
    t_start = time.perf_counter()
    body = await request.body()
    # Reconstruct float16 tensor from raw bytes
    tensor = torch.frombuffer(bytearray(body), dtype=torch.float16).reshape(1, 3, 224, 224)
    result = await BATCHER.predict(tensor)
    result["total_ms"] = round((time.perf_counter() - t_start) * 1000, 2)
    return JSONResponse(result)
