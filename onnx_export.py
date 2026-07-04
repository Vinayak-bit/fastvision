import torch
import timm
import time
import numpy as np
import pandas as pd
import copy
import json
import os
import onnxruntime as ort

def benchmark_pytorch(model, device, n_runs=200):
    model = model.to(device)
    dummy = torch.randn(1, 3, 224, 224).to(device)
    for _ in range(20):
        with torch.no_grad():
            model(dummy)
    if device.type == 'mps':
        torch.mps.synchronize()
    latencies = []
    for _ in range(n_runs):
        start = time.perf_counter()
        with torch.no_grad():
            model(dummy)
        if device.type == 'mps':
            torch.mps.synchronize()
        latencies.append((time.perf_counter() - start) * 1000)
    return {
        'p50': np.percentile(latencies, 50),
        'p95': np.percentile(latencies, 95),
        'p99': np.percentile(latencies, 99),
    }

def benchmark_onnx(session, n_runs=200):
    dummy = np.random.randn(1, 3, 224, 224).astype(np.float32)
    for _ in range(20):
        session.run(None, {'image': dummy})
    latencies = []
    for _ in range(n_runs):
        start = time.perf_counter()
        session.run(None, {'image': dummy})
        latencies.append((time.perf_counter() - start) * 1000)
    return {
        'p50': np.percentile(latencies, 50),
        'p95': np.percentile(latencies, 95),
        'p99': np.percentile(latencies, 99),
    }

def export_onnx(model, path):
    model_cpu = copy.deepcopy(model).to('cpu').eval()
    dummy = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        model_cpu,
        dummy,
        path,
        opset_version=18,       # match what new PyTorch actually has
        input_names=['image'],
        output_names=['logits'],
        dynamic_axes={
            'image':  {0: 'batch_size'},
            'logits': {0: 'batch_size'},
        }
    )
    print(f"Exported: {path}")

def load_onnx_session(path):
    # CoreML requires absolute path -- relative paths cause initializer error
    abs_path = os.path.abspath(path)
    available = ort.get_available_providers()
    # Try CoreML first (Apple Silicon native), fall back to CPU
    providers = [p for p in ['CoreMLExecutionProvider', 'CPUExecutionProvider'] if p in available]
    print(f"  Using ONNX providers: {providers}")
    try:
        sess = ort.InferenceSession(abs_path, providers=providers)
        return sess
    except Exception as e:
        print(f"  CoreML failed ({e}), falling back to CPU only")
        return ort.InferenceSession(abs_path, providers=['CPUExecutionProvider'])


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


# ── Main ──────────────────────────────────────────
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")

# ── Baseline ──────────────────────────────────────
print("\n--- Exporting BASELINE model ---")
baseline = timm.create_model('vit_base_patch16_224', pretrained=True)
baseline.eval()
export_onnx(baseline, 'onnx/baseline.onnx')
sess_baseline = load_onnx_session('onnx/baseline.onnx')
baseline_onnx = benchmark_onnx(sess_baseline)
print(f"Baseline ONNX  P50: {baseline_onnx['p50']:.1f}ms")
print(f"Baseline ONNX  P95: {baseline_onnx['p95']:.1f}ms")
print(f"Baseline ONNX  P99: {baseline_onnx['p99']:.1f}ms")

# ── Pruned 25% ────────────────────────────────────
print("\n--- Exporting PRUNED 25% model ---")
with open('results/head_importance.json') as f:
    importance = json.load(f)
importance = {int(k): v for k, v in importance.items()}
heads_to_prune = get_heads_to_prune(importance, ratio=0.25)

pruned = timm.create_model('vit_base_patch16_224', pretrained=False)
for layer_idx, heads in heads_to_prune.items():
    prune_attention_heads(pruned.blocks[layer_idx], set(heads))
pruned.load_state_dict(torch.load('checkpoints/pruned_25.pt', map_location='cpu'))
pruned.eval()

export_onnx(pruned, 'onnx/pruned_25.onnx')
sess_pruned = load_onnx_session('onnx/pruned_25.onnx')
pruned_onnx = benchmark_onnx(sess_pruned)
print(f"Pruned 25% ONNX P50: {pruned_onnx['p50']:.1f}ms")
print(f"Pruned 25% ONNX P95: {pruned_onnx['p95']:.1f}ms")
print(f"Pruned 25% ONNX P99: {pruned_onnx['p99']:.1f}ms")

# ── Update results.csv ─────────────────────────────
df = pd.read_csv('results/results.csv')
new_rows = pd.DataFrame([
    {'model': 'baseline_onnx', 'device': 'cpu',
     'p50_ms': baseline_onnx['p50'], 'p95_ms': baseline_onnx['p95'],
     'p99_ms': baseline_onnx['p99'], 'size_mb': None},
    {'model': 'pruned_25_onnx', 'device': 'cpu',
     'p50_ms': pruned_onnx['p50'], 'p95_ms': pruned_onnx['p95'],
     'p99_ms': pruned_onnx['p99'], 'size_mb': None},
])
df = pd.concat([df, new_rows], ignore_index=True)
df.to_csv('results/results.csv', index=False)

print("\n=== Full Results So Far ===")
print(df.to_string(index=False))
print("\nSaved: onnx/baseline.onnx, onnx/pruned_25.onnx, results/results.csv updated")
