import torch
import torch.nn.functional as F
import timm, time, numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

INET_MAP = {'n01440764':0,'n02102040':217,'n02979186':482,'n03000684':491,
            'n03028079':497,'n03394916':566,'n03417042':569,'n03425413':571,
            'n03445777':574,'n03888257':701}

DEV = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

def get_loaders(batch_train=32, batch_val=32):
    _t = transforms.Compose([
        transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    _v = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    tds = datasets.ImageFolder('data/imagenette2-320/train', transform=_t)
    vds = datasets.ImageFolder('data/imagenette2-320/val',   transform=_v)
    lmap = {i: INET_MAP[n] for i, n in enumerate(tds.classes)}
    tl = DataLoader(tds, batch_size=batch_train, shuffle=True,  num_workers=0)
    vl = DataLoader(vds, batch_size=batch_val,   shuffle=False, num_workers=0)
    return tl, vl, lmap

def evaluate(model, vloader, lmap):
    model.float().to(DEV).eval()
    ok = tot = 0
    with torch.inference_mode():
        for xx, yy in vloader:
            xx = xx.float().to(DEV)
            gt = torch.tensor([lmap[v.item()] for v in yy]).to(DEV)
            ok  += (model(xx).argmax(1) == gt).sum().item()
            tot += gt.size(0)
    model.float().to(DEV)
    return ok / tot

def benchmark_p99(model, n=200):
    snap = {k: v.clone() for k, v in model.state_dict().items()}
    model.half().to(DEV).eval()
    x = torch.randn(1, 3, 224, 224).half().to(DEV)
    for _ in range(50):
        with torch.inference_mode(): model(x)
    torch.mps.synchronize()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        with torch.inference_mode(): model(x)
        torch.mps.synchronize()
        ts.append((time.perf_counter() - t0) * 1000)
    out = float(np.percentile(ts, 99))
    model.float().to(DEV)
    model.load_state_dict({k: v.float() for k, v in snap.items()})
    return out

def cut_attn(blk, rm, hd=64):
    attn = blk.attn; onh = attn.num_heads; edim = attn.proj.in_features
    keep = [h for h in range(onh) if h not in rm]; nnh = len(keep)
    if nnh == onh: return
    ridx = []
    for sec in range(3):
        off = sec*onh*hd
        for h in keep: ridx.extend(range(off+h*hd, off+h*hd+hd))
    ridx = torch.tensor(ridx, dtype=torch.long)
    ow = attn.qkv.weight.data; ob = attn.qkv.bias.data if attn.qkv.bias is not None else None
    nw = ow[ridx,:].clone()
    nq = torch.nn.Linear(edim, nw.shape[0], bias=(ob is not None))
    nq.weight.data = nw
    if ob is not None: nq.bias.data = ob[ridx].clone()
    attn.qkv = nq
    cidx = []
    for h in keep: cidx.extend(range(h*hd, h*hd+hd))
    cidx = torch.tensor(cidx, dtype=torch.long)
    pw = attn.proj.weight.data; npw = pw[:,cidx].clone()
    np_ = torch.nn.Linear(npw.shape[1], edim, bias=(attn.proj.bias is not None))
    np_.weight.data = npw
    if attn.proj.bias is not None: np_.bias.data = attn.proj.bias.data.clone()
    attn.proj = np_; attn.num_heads = nnh; attn.attn_dim = nnh*hd

def cut_mlp(blk, rm):
    mlp = blk.mlp; old_n = mlp.fc1.out_features; edim = mlp.fc1.in_features
    keep = [n for n in range(old_n) if n not in rm]; new_n = len(keep)
    if new_n == old_n: return
    kidx = torch.tensor(keep, dtype=torch.long)
    ow1 = mlp.fc1.weight.data; ob1 = mlp.fc1.bias.data if mlp.fc1.bias is not None else None
    nf1 = torch.nn.Linear(edim, new_n, bias=(ob1 is not None))
    nf1.weight.data = ow1[kidx,:].clone()
    if ob1 is not None: nf1.bias.data = ob1[kidx].clone()
    mlp.fc1 = nf1
    ow2 = mlp.fc2.weight.data; ob2 = mlp.fc2.bias.data if mlp.fc2.bias is not None else None
    nf2 = torch.nn.Linear(new_n, edim, bias=(ob2 is not None))
    nf2.weight.data = ow2[:,kidx].clone()
    if ob2 is not None: nf2.bias.data = ob2.clone()
    mlp.fc2 = nf2

def taylor_scores_attn(model, ldr, lmap, nb=10):
    model.float().to(DEV).train()
    sc = {li: torch.zeros(model.blocks[li].attn.num_heads) for li in range(len(model.blocks))}
    it = iter(ldr)
    for i in range(nb):
        try:    xx, yy = next(it)
        except: it = iter(ldr); xx, yy = next(it)
        xx = xx.float().to(DEV)
        gt = torch.tensor([lmap[v.item()] for v in yy]).to(DEV)
        F.cross_entropy(model(xx), gt).backward()
        for li, blk in enumerate(model.blocks):
            g = blk.attn.qkv.weight.grad
            if g is None: continue
            nh = blk.attn.num_heads; hd = g.shape[0]//(3*nh)
            for h in range(nh):
                s = h*hd; w = blk.attn.qkv.weight.data[s:s+hd]
                sc[li][h] += (g[s:s+hd]*w).abs().mean().item()
        model.zero_grad()
        print(f"  attn score {i+1}/{nb}")
    model.eval()
    return {li: v.tolist() for li, v in sc.items()}

def taylor_scores_mlp(model, ldr, lmap, nb=10):
    model.float().to(DEV).train()
    sc = {li: torch.zeros(model.blocks[li].mlp.fc1.out_features) for li in range(len(model.blocks))}
    it = iter(ldr)
    for i in range(nb):
        try:    xx, yy = next(it)
        except: it = iter(ldr); xx, yy = next(it)
        xx = xx.float().to(DEV)
        gt = torch.tensor([lmap[v.item()] for v in yy]).to(DEV)
        F.cross_entropy(model(xx), gt).backward()
        for li, blk in enumerate(model.blocks):
            g = blk.mlp.fc1.weight.grad
            if g is None: continue
            w = blk.mlp.fc1.weight.data
            sc[li] += (g*w).abs().mean(dim=1).detach().cpu()
        model.zero_grad()
        print(f"  mlp score {i+1}/{nb}")
    model.eval()
    return {li: v.tolist() for li, v in sc.items()}

def which_heads(sc_dict, ratio):
    flat = [(s,li,hi) for li,sv in sc_dict.items() for hi,s in enumerate(sv)]
    flat.sort(key=lambda z: z[0])
    n = int(len(flat)*ratio); out = {}
    for _,li,hi in flat[:n]: out.setdefault(li,[]).append(hi)
    return out

def which_neurons(sc_dict, ratio):
    flat = [(s,li,ni) for li,sv in sc_dict.items() for ni,s in enumerate(sv)]
    flat.sort(key=lambda z: z[0])
    n = int(len(flat)*ratio); out = {}
    for _,li,ni in flat[:n]: out.setdefault(li,[]).append(ni)
    return out

def finetune(model, tloader, lmap, epochs=5, lr=1e-5):
    model.float().to(DEV)
    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best_acc = 0; best_state = None
    _, vl, _ = get_loaders()
    for ep in range(epochs):
        model.train(); rl = nb = 0
        for xx, yy in tloader:
            xx = xx.float().to(DEV)
            gt = torch.tensor([lmap[v.item()] for v in yy]).to(DEV)
            opt.zero_grad()
            loss = F.cross_entropy(model(xx), gt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); rl += loss.item(); nb += 1
        sched.step()
        va = evaluate(model, vl, lmap)
        print(f"  ep{ep+1}/{epochs}  loss={rl/nb:.4f}  acc={va*100:.1f}%")
        if va > best_acc:
            best_acc = va
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            print(f"    ^ best {best_acc*100:.1f}%")
    if best_state: model.load_state_dict(best_state)
    return model, best_acc
