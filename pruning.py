import torch
import torch.nn.functional as F
import timm
import time
import numpy as np
import pandas as pd
import copy
import json
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

IMAGENETTE_WNID_TO_IMAGENET_IDX = {
    'n01440764': 0, 'n02102040': 217, 'n02979186': 482, 'n03000684': 491,
    'n03028079': 497, 'n03394916': 566, 'n03417042': 569, 'n03425413': 571,
    'n03445777': 574, 'n03888257': 701,
}

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

def sanity_check(model, device, label=""):
    model = model.to(device)
    model.eval()
    dummy = torch.randn(1, 3, 224, 224).to(device)
    with torch.no_grad():
        out = model(dummy)
    has_nan = torch.isnan(out).any().item()
    print(f"[{label}] output shape: {tuple(out.shape)}, has_nan: {has_nan}, "
          f"min: {out.min().item():.2f}, max: {out.max().item():.2f}")
    return not has_nan


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


def prune_model(model, heads_to_prune_by_layer, head_dim=64):
    model = copy.deepcopy(model)
    for layer_idx, heads in heads_to_prune_by_layer.items():
        block = model.blocks[layer_idx]
        prune_attention_heads(block, set(heads), head_dim=head_dim)
    return model


# ── Main ──────────────────────────────────────────
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")

model = timm.create_model('vit_base_patch16_224', pretrained=True)
model.eval()

print("\nSanity-checking baseline model (pre-pruning)...")
sanity_check(model, device, label="baseline")

# ── Compute head importance using REAL ImageNette images + true labels ──
print("\nLoading real ImageNette training data for importance scoring...")
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
train_dataset = datasets.ImageFolder('data/imagenette2-320/train', transform=transform)
folder_idx_to_imagenet_idx = {
    i: IMAGENETTE_WNID_TO_IMAGENET_IDX[wnid]
    for i, wnid in enumerate(train_dataset.classes)
}
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)

def compute_head_importance(model, loader, device, folder_to_imagenet, n_batches=10):
    importance = {}
    model = model.to(device)
    model.train()

    for layer_idx, block in enumerate(model.blocks):
        n_heads = block.attn.num_heads
        scores = torch.zeros(n_heads)
        importance[layer_idx] = scores
    importance = {l: torch.zeros(model.blocks[l].attn.num_heads) for l in range(len(model.blocks))}

    data_iter = iter(loader)
    for i in range(n_batches):
        try:
            images, folder_labels = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            images, folder_labels = next(data_iter)

        images = images.to(device)
        true_labels = torch.tensor(
            [folder_to_imagenet[l.item()] for l in folder_labels]
        ).to(device)

        out = model(images)
        loss = F.cross_entropy(out, true_labels)
        loss.backward()

        for layer_idx, block in enumerate(model.blocks):
            grad = block.attn.qkv.weight.grad
            if grad is not None:
                n_heads = block.attn.num_heads
                head_dim = grad.shape[0] // (3 * n_heads)
                for h in range(n_heads):
                    start = h * head_dim
                    end = start + head_dim
                    importance[layer_idx][h] += grad[start:end].abs().mean().item()

        model.zero_grad()
        print(f"  batch {i+1}/{n_batches} processed (real images, true labels)")

    model.eval()
    return {l: v.tolist() for l, v in importance.items()}

print("\nComputing head importance (gradient-based, REAL data, ~2-3 min)...")
importance = compute_head_importance(model, train_loader, device, folder_idx_to_imagenet_idx, n_batches=10)

with open('results/head_importance.json', 'w') as f:
    json.dump(importance, f, indent=2)
print("Saved: results/head_importance.json (recomputed with real data)")

# ── Prune 50% ─────────────────────────────────────
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

print("\nPruning 50% of heads (using real-data importance scores)...")
heads_to_prune = get_heads_to_prune(importance, ratio=0.25)
print(f"Heads to prune by layer: {heads_to_prune}")

pruned_model = prune_model(model, heads_to_prune)
pruned_model.eval()

print("\nSanity-checking pruned model...")
ok = sanity_check(pruned_model, device, label="pruned_50")
if not ok:
    print("WARNING: pruned model produced NaN output! Do not trust benchmark.")

print("\nBenchmarking pruned model...")
pruned_results = benchmark(pruned_model, device)
pruned_size = get_model_size_mb(pruned_model)

print(f"\n=== Pruned 50% Results (real-data importance) ===")
print(f"P50: {pruned_results['p50']:.1f}ms")
print(f"P95: {pruned_results['p95']:.1f}ms")
print(f"P99: {pruned_results['p99']:.1f}ms")
print(f"Size: {pruned_size:.1f}MB")

torch.save(pruned_model.state_dict(), 'checkpoints/pruned_50.pt')

df = pd.read_csv('results/results.csv')
df = df[df['model'] != 'pruned_50_fp32']  # remove old (random-importance) result
new_row = pd.DataFrame([{
    'model': 'pruned_50_fp32',
    'device': str(device),
    'p50_ms': pruned_results['p50'],
    'p95_ms': pruned_results['p95'],
    'p99_ms': pruned_results['p99'],
    'size_mb': pruned_size,
}])
df = pd.concat([df, new_row], ignore_index=True)
df.to_csv('results/results.csv', index=False)

print("\n=== Full Results So Far ===")
print(df.to_string(index=False))
print("\nSaved: checkpoints/pruned_50.pt, results/results.csv updated")
