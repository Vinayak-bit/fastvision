"""
evaluate.py -- universal checkpoint evaluator for FastVision

Usage:
  python evaluate.py --model vit_base --checkpoint checkpoints/mlp_base_rnd3.pt
  python evaluate.py --model vit_small --checkpoint checkpoints/vit_small_finetuned_best.pt
  python evaluate.py --model vit_tiny  --checkpoint checkpoints/vit_tiny_finetuned_best.pt
  python evaluate.py --model vit_base  --checkpoint checkpoints/baseline.pt --no-prune

Flags:
  --model       vit_base | vit_small | vit_tiny
  --checkpoint  path to .pt file
  --no-prune    skip architecture reconstruction (for unpruned checkpoints)
  --n-warmup    warmup runs (default 50)
  --n-bench     benchmark runs (default 200)
  --accuracy    also run accuracy evaluation on imagenette val set
"""

import argparse
import torch
import timm
import time
import numpy as np

# ── args ──────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--model',      required=True, choices=['vit_base','vit_small','vit_tiny'])
parser.add_argument('--checkpoint', required=True)
parser.add_argument('--no-prune',   action='store_true')
parser.add_argument('--n-warmup',   type=int, default=50)
parser.add_argument('--n-bench',    type=int, default=200)
parser.add_argument('--accuracy',   action='store_true')
args = parser.parse_args()

MODEL_NAMES = {
    'vit_base':  'vit_base_patch16_224',
    'vit_small': 'vit_small_patch16_224',
    'vit_tiny':  'vit_tiny_patch16_224',
}
HEAD_DIMS = {
    'vit_base': 64,
    'vit_small': 64,
    'vit_tiny': 64,
}

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"device:     {device}")
print(f"model:      {args.model}")
print(f"checkpoint: {args.checkpoint}")
print(f"pytorch:    {torch.__version__}")

# ── load checkpoint ───────────────────────────────
state = torch.load(args.checkpoint, map_location='cpu')
hd    = HEAD_DIMS[args.model]

# ── build architecture ────────────────────────────
model = timm.create_model(MODEL_NAMES[args.model], pretrained=False)

if not args.no_prune:
    # reconstruct pruned attention layers from checkpoint shapes
    for li in range(len(model.blocks)):
        key = f'blocks.{li}.attn.qkv.weight'
        if key in state:
            new_nh = state[key].shape[0] // (3*hd)
            old_nh = model.blocks[li].attn.num_heads
            if new_nh != old_nh:
                attn = model.blocks[li].attn
                edim = attn.proj.in_features
                keep_idx = []
                for sec in range(3):
                    off = sec*old_nh*hd
                    for h in range(new_nh):
                        keep_idx.extend(range(off+h*hd, off+h*hd+hd))
                keep_idx = torch.tensor(keep_idx, dtype=torch.long)
                nq = torch.nn.Linear(edim, len(keep_idx), bias=True)
                attn.qkv = nq
                cidx = []
                for h in range(new_nh):
                    cidx.extend(range(h*hd, h*hd+hd))
                cidx = torch.tensor(cidx, dtype=torch.long)
                np_ = torch.nn.Linear(len(cidx), edim, bias=True)
                attn.proj = np_
                attn.num_heads = new_nh
                attn.attn_dim  = new_nh * hd

    # reconstruct pruned MLP layers from checkpoint shapes
    for li in range(len(model.blocks)):
        key = f'blocks.{li}.mlp.fc1.weight'
        if key in state:
            new_n = state[key].shape[0]
            old_n = model.blocks[li].mlp.fc1.out_features
            edim  = model.blocks[li].mlp.fc1.in_features
            if new_n != old_n:
                model.blocks[li].mlp.fc1 = torch.nn.Linear(edim, new_n, bias=True)
                model.blocks[li].mlp.fc2 = torch.nn.Linear(new_n, edim, bias=True)

# ── load weights ──────────────────────────────────
model.load_state_dict({k: v.half() for k, v in state.items()})
model = model.half().to(device).eval()

# print architecture summary
heads   = [b.attn.num_heads for b in model.blocks]
neurons = [b.mlp.fc1.out_features for b in model.blocks]
print(f"attn heads:  {heads}")
print(f"mlp neurons: {neurons}")

# ── warmup ────────────────────────────────────────
dummy = torch.randn(1, 3, 224, 224).half().to(device)
print(f"\nwarming up ({args.n_warmup} runs)...")
for _ in range(args.n_warmup):
    with torch.inference_mode(): model(dummy)
torch.mps.synchronize()
print("warmup done")

# ── benchmark ─────────────────────────────────────
latencies = []
for _ in range(args.n_bench):
    t = time.perf_counter()
    with torch.inference_mode(): model(dummy)
    torch.mps.synchronize()
    latencies.append((time.perf_counter() - t) * 1000)

print(f"\n=== Latency ({args.n_bench} runs) ===")
print(f"P50: {np.percentile(latencies,50):.1f}ms")
print(f"P95: {np.percentile(latencies,95):.1f}ms")
print(f"P99: {np.percentile(latencies,99):.1f}ms")
print(f"Min: {np.min(latencies):.1f}ms")
print(f"Max: {np.max(latencies):.1f}ms")

# ── accuracy (optional) ───────────────────────────
if args.accuracy:
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader

    INET_MAP = {
        'n01440764':0,'n02102040':217,'n02979186':482,'n03000684':491,
        'n03028079':497,'n03394916':566,'n03417042':569,'n03425413':571,
        'n03445777':574,'n03888257':701,
    }
    val_tfm = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    vds    = datasets.ImageFolder('data/imagenette2-320/val', transform=val_tfm)
    lmap   = {i: INET_MAP[n] for i, n in enumerate(vds.classes)}
    vloader = DataLoader(vds, batch_size=32, shuffle=False, num_workers=0)

    model.float().to(device).eval()
    ok = tot = 0
    with torch.inference_mode():
        for imgs, lbls in vloader:
            imgs = imgs.float().to(device)
            gt   = torch.tensor([lmap[l.item()] for l in lbls]).to(device)
            ok  += (model(imgs).argmax(1) == gt).sum().item()
            tot += gt.size(0)
    print(f"\n=== Accuracy ===")
    print(f"{ok/tot*100:.1f}% on {tot} ImageNette val images")
