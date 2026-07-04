import torch
import torch.nn.functional as F
import timm
import time
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

IMAGENETTE_MAP = {
    'n01440764': 0, 'n02102040': 217, 'n02979186': 482, 'n03000684': 491,
    'n03028079': 497, 'n03394916': 566, 'n03417042': 569, 'n03425413': 571,
    'n03445777': 574, 'n03888257': 701,
}

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")

transform_train = transforms.Compose([
    transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])
transform_val = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

train_dataset = datasets.ImageFolder('data/imagenette2-320/train', transform=transform_train)
val_dataset   = datasets.ImageFolder('data/imagenette2-320/val',   transform=transform_val)
folder_to_imagenet = {i: IMAGENETTE_MAP[w] for i, w in enumerate(train_dataset.classes)}

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_dataset,   batch_size=32, shuffle=False, num_workers=0)

def evaluate(model):
    # Always FP32 eval -- no dtype confusion
    model.float().to(device).eval()
    correct, total = 0, 0
    with torch.inference_mode():
        for images, folder_labels in val_loader:
            images = images.float().to(device)
            true_labels = torch.tensor(
                [folder_to_imagenet[l.item()] for l in folder_labels]
            ).to(device)
            preds = model(images).argmax(dim=1)
            correct += (preds == true_labels).sum().item()
            total += true_labels.size(0)
    # Restore to FP32 training mode after eval
    model.float().to(device)
    return correct / total

def benchmark_p99(model, n=200):
    # Benchmark in FP16, then restore to FP32 for training
    saved_state = {k: v.clone() for k, v in model.state_dict().items()}
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
    p99 = np.percentile(latencies, 99)
    # Restore model to FP32 for continued training
    model.float().to(device)
    model.load_state_dict({k: v.float() for k, v in saved_state.items()})
    return p99

# ── Load pretrained ViT-Tiny ───────────────────────
print("Loading ViT-Tiny pretrained...")
model = timm.create_model('vit_tiny_patch16_224', pretrained=True)
model = model.float().to(device)

acc_before = evaluate(model)
p99_before = benchmark_p99(model)
print(f"Before: {acc_before*100:.1f}% acc, {p99_before:.1f}ms P99")

# ── Fine-tune ──────────────────────────────────────
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

best_acc = acc_before
best_state = None

print("\nFine-tuning ViT-Tiny (10 epochs)...")
for epoch in range(10):
    # Ensure model is in FP32 train mode at start of each epoch
    model.float().to(device).train()
    total_loss, n_batches = 0, 0

    for images, folder_labels in train_loader:
        images = images.float().to(device)
        true_labels = torch.tensor(
            [folder_to_imagenet[l.item()] for l in folder_labels]
        ).to(device)
        optimizer.zero_grad()
        loss = F.cross_entropy(model(images), true_labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1

    scheduler.step()
    val_acc = evaluate(model)     # FP32 eval, restores FP32 after
    p99 = benchmark_p99(model)    # FP16 bench, restores FP32 after
    print(f"  Epoch {epoch+1}/10 | loss={total_loss/n_batches:.4f} | acc={val_acc*100:.1f}% | P99={p99:.1f}ms")

    # Save every epoch
    torch.save(model.state_dict(), f'checkpoints/vit_tiny_finetuned_epoch{epoch+1}.pt')

    if val_acc > best_acc:
        best_acc = val_acc
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
        torch.save(best_state, 'checkpoints/vit_tiny_finetuned_best.pt')
        print(f"    ↑ New best: {best_acc*100:.1f}%")

# ── Final results ──────────────────────────────────
if best_state:
    model.load_state_dict({k: v.float() for k, v in best_state.items()})

final_acc = evaluate(model)
final_p99 = benchmark_p99(model)

print(f"\n=== ViT-Tiny Fine-tuning Complete ===")
print(f"Before: {acc_before*100:.1f}% acc  {p99_before:.1f}ms P99")
print(f"After:  {final_acc*100:.1f}% acc  {final_p99:.1f}ms P99")

print(f"\n=== Complete Tradeoff Curve (MPS FP16) ===")
print(f"{'Model':<35} {'P99':>8} {'Accuracy':>10}")
print(f"{'-'*55}")
print(f"{'ViT-Base FP32 (baseline)':<35} {'24.6ms':>8} {'97.2%':>10}")
print(f"{'ViT-Base FP16':<35} {'24.0ms':>8} {'97.2%':>10}")
print(f"{'ViT-Base pruned 25% FP16':<35} {'20.9ms':>8} {'91.9%':>10}")
print(f"{'ViT-Small pretrained FP16':<35} {'7.4ms':>8}  {'76.6%':>10}")
print(f"{'ViT-Small finetuned FP16':<35} {'12.9ms':>8} {'99.6%':>10}")
print(f"{'ViT-Small pruned 10% FP16':<35} {'13.3ms':>8} {'98.6%':>10}")
print(f"{'ViT-Tiny pretrained FP16':<35} {p99_before:>7.1f}ms {acc_before*100:>9.1f}%")
print(f"{'ViT-Tiny finetuned FP16':<35} {final_p99:>7.1f}ms {final_acc*100:>9.1f}%")
