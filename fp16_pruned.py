import torch
import timm
import time
import numpy as np
import pandas as pd
import json

def benchmark(model, device, n_runs=200, dtype=torch.float32):
    model = model.to(device)
    dummy = torch.randn(1, 3, 224, 224, dtype=dtype).to(device)
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

def get_model_size_mb(model):
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix='.pt') as f:
        torch.save(model.state_dict(), f.name)
        return os.path.getsize(f.name) / 1e6

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

# ── Setup ──────────────────────────────────────────
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")

with open('results/head_importance.json') as f:
    importance = json.load(f)
importance = {int(k): v for k, v in importance.items()}
heads_to_prune = get_heads_to_prune(importance, ratio=0.25)

# ── Build pruned model, cast to FP16 ──────────────
print("\nBuilding pruned 25% + FP16 model...")
model = timm.create_model('vit_base_patch16_224', pretrained=False)
for layer_idx, heads in heads_to_prune.items():
    prune_attention_heads(model.blocks[layer_idx], set(heads))
model.load_state_dict(torch.load('checkpoints/pruned_25.pt', map_location='cpu'))
model.eval()

# Cast entire model to FP16 then move to MPS
model_fp16 = model.half().to(device)
size = get_model_size_mb(model_fp16)

print("Benchmarking pruned 25% + FP16 on MPS...")
results = benchmark(model_fp16, device, dtype=torch.float16)

print(f"\n=== Pruned 25% + FP16 Results ===")
print(f"P50: {results['p50']:.1f}ms")
print(f"P95: {results['p95']:.1f}ms")
print(f"P99: {results['p99']:.1f}ms")
print(f"Size: {size:.1f}MB")

torch.save(model_fp16.state_dict(), 'checkpoints/pruned_25_fp16.pt')

# ── Full comparison table ──────────────────────────
df = pd.read_csv('results/results.csv')
new_row = pd.DataFrame([{
    'model': 'pruned_25_fp16_mps',
    'device': str(device),
    'p50_ms': results['p50'],
    'p95_ms': results['p95'],
    'p99_ms': results['p99'],
    'size_mb': size,
}])
df = pd.concat([df, new_row], ignore_index=True)
df.to_csv('results/results.csv', index=False)

print("\n=== Full Results So Far ===")
# Only show MPS rows for clean comparison
mps_df = df[df['device'] == str(device)]
print(mps_df.to_string(index=False))

print(f"\nSLA target: P99 < 10ms")
print(f"Current best: {df[df['device']==str(device)]['p99_ms'].min():.1f}ms")
print(f"Gap remaining: {df[df['device']==str(device)]['p99_ms'].min() - 10:.1f}ms")
