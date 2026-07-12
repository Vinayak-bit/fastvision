import torch
import torch.nn.functional as F
import timm, time, numpy as np, json
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

INET_MAP = {'n01440764':0,'n02102040':217,'n02979186':482,'n03000684':491,
            'n03028079':497,'n03394916':566,'n03417042':569,'n03425413':571,
            'n03445777':574,'n03888257':701}

dev = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"device: {dev}")

_tds = datasets.ImageFolder('data/imagenette2-320/train', transform=transforms.Compose([
    transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(),
    transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])]))
_vds = datasets.ImageFolder('data/imagenette2-320/val', transform=transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224),
    transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])]))
lmap    = {i: INET_MAP[n] for i, n in enumerate(_tds.classes)}
tloader = DataLoader(_tds, batch_size=16, shuffle=True,  num_workers=0)
vloader = DataLoader(_vds, batch_size=32, shuffle=False, num_workers=0)


def acc_check(mdl):
    mdl.float().to(dev).eval()
    ok = tot = 0
    with torch.inference_mode():
        for xx, yy in vloader:
            xx = xx.float().to(dev)
            gt = torch.tensor([lmap[v.item()] for v in yy]).to(dev)
            ok  += (mdl(xx).argmax(1) == gt).sum().item()
            tot += gt.size(0)
    mdl.float().to(dev)
    return ok / tot


def mlp_scores(mdl, ldr, nb=10):
    mdl.float().to(dev).train()
    sc = {li: torch.zeros(mdl.blocks[li].mlp.fc1.out_features)
          for li in range(len(mdl.blocks))}
    it = iter(ldr)
    for i in range(nb):
        try:    xx, yy = next(it)
        except: it = iter(ldr); xx, yy = next(it)
        xx = xx.float().to(dev)
        gt = torch.tensor([lmap[v.item()] for v in yy]).to(dev)
        F.cross_entropy(mdl(xx), gt).backward()
        for li, blk in enumerate(mdl.blocks):
            g = blk.mlp.fc1.weight.grad
            if g is None: continue
            w = blk.mlp.fc1.weight.data
            sc[li] += (g * w).abs().mean(dim=1).detach().cpu()
        mdl.zero_grad()
        print(f"  mlp score batch {i+1}/{nb}")
    mdl.eval()
    return {li: v.tolist() for li, v in sc.items()}


def which_neurons(sc_dict, ratio):
    flat = [(s, li, ni) for li, sv in sc_dict.items()
            for ni, s in enumerate(sv)]
    flat.sort(key=lambda z: z[0])
    n = int(len(flat) * ratio)
    out = {}
    for _, li, ni in flat[:n]:
        out.setdefault(li, []).append(ni)
    return out


def cut_mlp(blk, rm):
    mlp   = blk.mlp
    old_n = mlp.fc1.out_features
    edim  = mlp.fc1.in_features
    keep  = [n for n in range(old_n) if n not in rm]
    new_n = len(keep)
    if new_n == old_n: return
    kidx = torch.tensor(keep, dtype=torch.long)

    ow1 = mlp.fc1.weight.data
    ob1 = mlp.fc1.bias.data if mlp.fc1.bias is not None else None
    nw1 = ow1[kidx, :].clone()
    nf1 = torch.nn.Linear(edim, new_n, bias=(ob1 is not None))
    nf1.weight.data = nw1
    if ob1 is not None: nf1.bias.data = ob1[kidx].clone()
    mlp.fc1 = nf1

    ow2 = mlp.fc2.weight.data
    ob2 = mlp.fc2.bias.data if mlp.fc2.bias is not None else None
    nw2 = ow2[:, kidx].clone()
    nf2 = torch.nn.Linear(new_n, edim, bias=(ob2 is not None))
    nf2.weight.data = nw2
    if ob2 is not None: nf2.bias.data = ob2.clone()
    mlp.fc2 = nf2


def do_mlp_prune(mdl, rm_dict):
    for li, neurons in rm_dict.items():
        cut_mlp(mdl.blocks[li], set(neurons))


def finetune_quick(mdl, epochs=2, lr=1e-5):
    mdl.float().to(dev)
    opt   = torch.optim.AdamW(mdl.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    for ep in range(epochs):
        mdl.train()
        rl = nb = 0
        for xx, yy in tloader:
            xx = xx.float().to(dev)
            gt = torch.tensor([lmap[v.item()] for v in yy]).to(dev)
            opt.zero_grad()
            loss = F.cross_entropy(mdl(xx), gt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(mdl.parameters(), 1.0)
            opt.step()
            rl += loss.item(); nb += 1
        sched.step()
        va = acc_check(mdl)
        print(f"    ep{ep+1}/{epochs}  loss={rl/nb:.4f}  acc={va*100:.1f}%")
    mdl.float().to(dev)


# ── load iterative pruned ViT-Base (round 5) ──────
print("\nloading iterative pruned vit-base (round 5)...")
state = torch.load('checkpoints/iterative_rnd5.pt', map_location='cpu')

# reconstruct pruned attention architecture from weight shapes
model = timm.create_model('vit_base_patch16_224', pretrained=False)
head_dim = 64
for li in range(12):
    qkv_key = f'blocks.{li}.attn.qkv.weight'
    if qkv_key in state:
        new_nh = state[qkv_key].shape[0] // (3 * head_dim)
        old_nh = model.blocks[li].attn.num_heads
        if new_nh != old_nh:
            # rebuild attention with correct head count
            attn = model.blocks[li].attn
            edim = attn.proj.in_features
            keep = list(range(new_nh))
            keep_idx = []
            for sec in range(3):
                off = sec * old_nh * head_dim
                for h in keep:
                    keep_idx.extend(range(off+h*head_dim, off+h*head_dim+head_dim))
            keep_idx = torch.tensor(keep_idx, dtype=torch.long)
            nq = torch.nn.Linear(edim, len(keep_idx), bias=True)
            attn.qkv = nq
            cidx = []
            for h in keep:
                cidx.extend(range(h*head_dim, h*head_dim+head_dim))
            cidx = torch.tensor(cidx, dtype=torch.long)
            np_ = torch.nn.Linear(len(cidx), edim, bias=True)
            attn.proj = np_
            attn.num_heads = new_nh
            attn.attn_dim  = new_nh * head_dim

state_fp32 = {k: v.float() for k, v in state.items()}
model.load_state_dict(state_fp32)
model = model.float().to(dev)

a0 = acc_check(model)
print(f"starting point (iterative attn pruned): {a0*100:.1f}% acc")
print(f"MLP intermediate neurons per block: {model.blocks[0].mlp.fc1.out_features}")

# ── iterative MLP pruning ──────────────────────────
ROUNDS     = 3
RATIO_STEP = 0.10
results    = [(0, 0, a0)]  # p99 unreliable in-session, track acc only

for rnd in range(1, ROUNDS + 1):
    print(f"\n=== MLP round {rnd}/{ROUNDS} ===")
    n_total  = sum(b.mlp.fc1.out_features for b in model.blocks)
    n_remove = int(n_total * RATIO_STEP)
    print(f"  neurons remaining: {n_total}  removing: {n_remove}")

    print("  scoring mlp neurons (taylor)...")
    sc = mlp_scores(model, tloader, nb=10)

    rm = which_neurons(sc, ratio=RATIO_STEP)
    do_mlp_prune(model, rm)

    n_after = sum(b.mlp.fc1.out_features for b in model.blocks)
    print(f"  neurons after pruning: {n_after}")

    print("  finetuning 2 epochs...")
    finetune_quick(model, epochs=2, lr=1e-5)

    va = acc_check(model)
    print(f"  after round {rnd}: {va*100:.1f}% acc")

    torch.save(model.state_dict(),
               f'checkpoints/mlp_base_rnd{rnd}.pt')
    print(f"  saved: mlp_base_rnd{rnd}.pt")
    results.append((rnd, 0, va))

print(f"\n{'='*55}")
print(f"=== combined attn+mlp iterative pruning ===")
print(f"note: run clean benchmark after for real p99")
print(f"{'round':<10} {'accuracy':>10}")
print(f"{'-'*55}")
for rnd, _, acc in results:
    label = 'attn only' if rnd == 0 else f'mlp rnd {rnd}'
    print(f"{label:<10} {acc*100:>9.1f}%")
print(f"\nrun clean benchmark on checkpoints/mlp_base_rnd3.pt")
