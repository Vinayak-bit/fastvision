import torch
import torch.nn.functional as F
import timm
import time
import numpy as np
import json
import copy
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

IMAGENETTE_MAP = {
    'n01440764': 0, 'n02102040': 217, 'n02979186': 482, 'n03000684': 491,
    'n03028079': 497, 'n03394916': 566, 'n03417042': 569, 'n03425413': 571,
    'n03445777': 574, 'n03888257': 701,
}

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")

transform_val = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])
transform_train = transforms.Compose([
    transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

train_dataset = datasets.ImageFolder('data/imagenette2-320/train', transform=transform_train)
val_dataset   = datasets.ImageFolder('data/imagenette2-320/val',   transform=transform_val)
folder_to_imagenet = {i: IMAGENETTE_MAP[w] for i, w in enumerate(val_dataset.classes)}

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_dataset,   batch_size=32, shuffle=False, num_workers=0)

def evaluate(model):
    # Always force FP32 for accuracy eval -- avoids any dtype mismatch
    # between model weights and input images regardless of how model was loaded
    model_fp32 = model.float().to(device).eval()
    correct, total = 0, 0
    with torch.inference_mode():
        for images, folder_labels in val_loader:
            images = images.float().to(device)  # FP32 images
            true_labels = torch.tensor(
                [folder_to_imagenet[l.item()] for l in folder_labels]
            ).to(device)
            preds = model_fp32(images).argmax(dim=1)
            correct += (preds == true_labels).sum().item()
            total += true_labels.size(0)
    return correct / total

def benchmark_p99(model, n=200):
    # Always FP16 for latency benchmark
    model_fp16 = model.half().to(device).eval()
    dummy = torch.randn(1, 3, 224, 224).half().to(device)
    for _ in range(50):
        with torch.inference_mode(): model_fp16(dummy)
    torch.mps.synchronize()
    latencies = []
    for _ in range(n):
        t = time.perf_counter()
        with torch.inference_mode(): model_fp16(dummy)
        torch.mps.synchronize()
        latencies.append((time.perf_counter() - t) * 1000)
    return np.percentile(latencies, 99)

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
    old_qkv_bias   = attn.qkv.bias.data if attn.qkv.bias is not None else None
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
    attn.attn_dim  = new_num_heads * head_dim

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

def compute_head_importance(model, loader, device, n_batches=10):
    # Importance computation always in FP32 (gradients need full precision)
    model_fp32 = model.float().to(device)
    model_fp32.train()
    importance = {l: torch.zeros(model_fp32.blocks[l].attn.num_heads)
                  for l in range(len(model_fp32.blocks))}
    data_iter = iter(loader)
    for i in range(n_batches):
        try:
            images, folder_labels = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            images, folder_labels = next(data_iter)
        images = images.float().to(device)
        true_labels = torch.tensor(
            [folder_to_imagenet[l.item()] for l in folder_labels]
        ).to(device)
        out  = model_fp32(images)
        loss = F.cross_entropy(out, true_labels)
        loss.backward()
        for layer_idx, block in enumerate(model_fp32.blocks):
            grad = block.attn.qkv.weight.grad
            if grad is not None:
                n_heads  = block.attn.num_heads
                head_dim = grad.shape[0] // (3 * n_heads)
                for h in range(n_heads):
                    start = h * head_dim
                    importance[layer_idx][h] += grad[start:start+head_dim].abs().mean().item()
        model_fp32.zero_grad()
        print(f"  importance batch {i+1}/{n_batches}")
    model_fp32.eval()
    return {l: v.tolist() for l, v in importance.items()}

# ── Load finetuned ViT-Small in FP32 ──────────────
print("\nLoading ViT-Small finetuned (FP32)...")
base_model = timm.create_model('vit_small_patch16_224', pretrained=False)
state = torch.load('checkpoints/vit_small_finetuned_best.pt', map_location='cpu')
# Force all weights to FP32 on load -- checkpoint may have mixed dtypes
state = {k: v.float() for k, v in state.items()}
base_model.load_state_dict(state)
base_model = base_model.float().to(device)

print("Evaluating baseline accuracy...")
acc_base = evaluate(base_model)
print(f"Baseline: {acc_base*100:.1f}%")

p99_base = benchmark_p99(base_model)
print(f"Baseline P99 (FP16): {p99_base:.1f}ms")

# ── Compute importance ─────────────────────────────
print("\nComputing head importance...")
importance = compute_head_importance(base_model, train_loader, device, n_batches=10)
with open('results/head_importance_small.json', 'w') as f:
    json.dump(importance, f, indent=2)

# ── Test pruning ratios ────────────────────────────
print("\nTesting pruning ratios...")
results = []

for ratio in [0.30]:
    heads_to_prune = get_heads_to_prune(importance, ratio)
    pruned = copy.deepcopy(base_model)
    for layer_idx, heads in heads_to_prune.items():
        prune_attention_heads(pruned.blocks[layer_idx], set(heads))

    acc = evaluate(pruned)   # FP32 eval
    p99 = benchmark_p99(pruned)  # FP16 latency

    drop = (acc_base - acc) * 100
    flag = '✅' if acc >= 0.95 else '❌'
    print(f"  ratio={ratio:.0%}: P99={p99:.1f}ms  acc={acc*100:.1f}%  drop={drop:.1f}pts  {flag}")
    results.append((ratio, p99, acc))

    if acc >= 0.95:
        save_path = f'checkpoints/vit_small_pruned_{int(ratio*100)}_fp16.pt'
        torch.save(pruned.half().state_dict(), save_path)
        print(f"    Saved: {save_path}")

# ── Summary ────────────────────────────────────────
print(f"\n=== ViT-Small Pruning Summary ===")
print(f"{'Model':<35} {'P99':>8} {'Accuracy':>10} {'Drop':>8}")
print(f"{'-'*65}")
print(f"{'ViT-Small finetuned (no prune)':<35} {p99_base:>7.1f}ms {acc_base*100:>9.1f}% {'—':>8}")
for ratio, p99, acc in results:
    drop = (acc_base - acc) * 100
    flag = '✅' if acc >= 0.95 else '❌'
    name = f'ViT-Small pruned {ratio:.0%} FP16'
    print(f"{name:<35} {p99:>7.1f}ms {acc*100:>9.1f}% {drop:>7.1f}pts  {flag}")

print(f"\n--- Context ---")
print(f"ViT-Base FP16 baseline:         24.0ms    97.2%")
print(f"ViT-Base pruned 25% FP16:       20.9ms    91.9%")
print(f"ViT-Small FP16 pretrained:       7.4ms    76.6%")
print(f"ViT-Small FP16 finetuned:      {p99_base:.1f}ms    {acc_base*100:.1f}%")
