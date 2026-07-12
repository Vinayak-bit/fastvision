import torch
import torch.nn.functional as F
import timm, time, numpy as np, json, copy
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
tloader = DataLoader(_tds, batch_size=32, shuffle=True,  num_workers=0)
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


def get_p99(mdl, n=200):
    snap = {k: v.clone() for k, v in mdl.state_dict().items()}
    mdl.half().to(dev).eval()
    x = torch.randn(1, 3, 224, 224).half().to(dev)
    for _ in range(50):
        with torch.inference_mode(): mdl(x)
    torch.mps.synchronize()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        with torch.inference_mode(): mdl(x)
        torch.mps.synchronize()
        ts.append((time.perf_counter() - t0) * 1000)
    out = float(np.percentile(ts, 99))
    mdl.float().to(dev)
    mdl.load_state_dict({k: v.float() for k, v in snap.items()})
    return out


def taylor_scores(mdl, ldr, nb=10):
    # recompute importance on CURRENT model state
    # this is what makes iterative different from one-shot --
    # after each prune+finetune cycle the surviving heads have
    # adapted, so importance scores reflect the current model
    # not the original one
    mdl.float().to(dev).train()
    sc = {li: torch.zeros(mdl.blocks[li].attn.num_heads)
          for li in range(len(mdl.blocks))}
    it = iter(ldr)
    for _ in range(nb):
        try:    xx, yy = next(it)
        except: it = iter(ldr); xx, yy = next(it)
        xx = xx.float().to(dev)
        gt = torch.tensor([lmap[v.item()] for v in yy]).to(dev)
        F.cross_entropy(mdl(xx), gt).backward()
        for li, blk in enumerate(mdl.blocks):
            g = blk.attn.qkv.weight.grad
            if g is None: continue
            nh = blk.attn.num_heads
            hd = g.shape[0] // (3 * nh)
            for h in range(nh):
                s = h * hd
                w = blk.attn.qkv.weight.data[s:s+hd]
                sc[li][h] += (g[s:s+hd] * w).abs().mean().item()
        mdl.zero_grad()
    mdl.eval()
    return {li: v.tolist() for li, v in sc.items()}


def which_heads(sc_dict, ratio):
    flat = [(s, li, hi) for li, sv in sc_dict.items()
            for hi, s in enumerate(sv)]
    flat.sort(key=lambda z: z[0])
    n = int(len(flat) * ratio)
    out = {}
    for _, li, hi in flat[:n]:
        out.setdefault(li, []).append(hi)
    return out


def cutblk(blk, rm, hd=64):
    attn = blk.attn
    onh  = attn.num_heads
    edim = attn.proj.in_features
    keep = [h for h in range(onh) if h not in rm]
    nnh  = len(keep)
    if nnh == onh: return
    ridx = []
    for sec in range(3):
        off = sec * onh * hd
        for h in keep: ridx.extend(range(off+h*hd, off+h*hd+hd))
    ridx = torch.tensor(ridx, dtype=torch.long)
    ow = attn.qkv.weight.data
    ob = attn.qkv.bias.data if attn.qkv.bias is not None else None
    nw = ow[ridx, :].clone()
    nq = torch.nn.Linear(edim, nw.shape[0], bias=(ob is not None))
    nq.weight.data = nw
    if ob is not None: nq.bias.data = ob[ridx].clone()
    attn.qkv = nq
    cidx = []
    for h in keep: cidx.extend(range(h*hd, h*hd+hd))
    cidx = torch.tensor(cidx, dtype=torch.long)
    pw  = attn.proj.weight.data
    npw = pw[:, cidx].clone()
    np_ = torch.nn.Linear(npw.shape[1], edim,
                           bias=(attn.proj.bias is not None))
    np_.weight.data = npw
    if attn.proj.bias is not None:
        np_.bias.data = attn.proj.bias.data.clone()
    attn.proj = np_
    attn.num_heads = nnh
    attn.attn_dim  = nnh * hd


def do_prune(mdl, rm_dict):
    for li, hs in rm_dict.items():
        cutblk(mdl.blocks[li], set(hs))


def finetune_quick(mdl, epochs=2, lr=1e-5):
    # short finetune between each pruning step
    # goal: let surviving heads adapt before next importance scoring
    # 2 epochs is enough to stabilize without wasting hours
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


# ── main ──────────────────────────────────────────
print("\nloading vit-base...")
mdl = timm.create_model('vit_base_patch16_224', pretrained=True)
mdl = mdl.float().to(dev)

a0 = acc_check(mdl)
p0 = get_p99(mdl)
print(f"baseline: {a0*100:.1f}% acc  {p0:.1f}ms p99")

# iterative pruning:
# 5 rounds × 5% per round = 25% total removed
# each round: score → prune 5% → finetune 2 epochs
# key difference from one-shot: importance scores recomputed
# on the adapted model after each round, not original model
ROUNDS     = 5
RATIO_STEP = 0.05   # remove 5% of REMAINING heads each round
results    = [(0, p0, a0)]

for rnd in range(1, ROUNDS + 1):
    print(f"\n=== round {rnd}/{ROUNDS} ===")
    n_total   = sum(b.attn.num_heads for b in mdl.blocks)
    n_remove  = int(n_total * RATIO_STEP)
    print(f"  heads remaining: {n_total}  removing: {n_remove}")

    # recompute importance on current model -- this is the key step
    print("  scoring heads (taylor, current model)...")
    sc = taylor_scores(mdl, tloader, nb=10)

    # prune n_remove least important heads
    rm = which_heads(sc, ratio=RATIO_STEP)
    do_prune(mdl, rm)   # modifies mdl in place

    # quick finetune to let surviving heads adapt
    print("  finetuning 2 epochs...")
    finetune_quick(mdl, epochs=2, lr=1e-5)

    va = acc_check(mdl)
    vp = get_p99(mdl)
    print(f"  after round {rnd}: {va*100:.1f}% acc  {vp:.1f}ms p99")

    # save each round separately -- never lose progress
    torch.save(mdl.state_dict(),
               f'checkpoints/iterative_rnd{rnd}.pt')
    print(f"  saved: iterative_rnd{rnd}.pt")

    results.append((rnd, vp, va))

# final summary
print(f"\n{'='*55}")
print(f"=== iterative pruning results (5% x 5 rounds = 25%) ===")
print(f"{'round':<8} {'p99':>8} {'accuracy':>10} {'vs baseline':>12}")
print(f"{'-'*55}")
for rnd, p99, acc in results:
    label = 'baseline' if rnd == 0 else f'round {rnd}'
    gain  = p0 - p99
    print(f"{label:<8} {p99:>7.1f}ms {acc*100:>9.1f}%  {gain:>+10.1f}ms")

print(f"\nvs one-shot pruning:")
print(f"  one-shot 25% Taylor+FT:  22.2ms  99.8%")
print(f"  iterative 25% Taylor+FT: {results[-1][1]:.1f}ms  {results[-1][2]*100:.1f}%")
