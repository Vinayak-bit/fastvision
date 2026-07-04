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

# ── Data ───────────────────────────────────────────
transform_train = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
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

def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.inference_mode():
        for images, folder_labels in loader:
            images = images.to(device)
            true_labels = torch.tensor(
                [folder_to_imagenet[l.item()] for l in folder_labels]
            ).to(device)
            preds = model(images).argmax(dim=1)
            correct += (preds == true_labels).sum().item()
            total += true_labels.size(0)
    return correct / total

def benchmark_p99(model, device, n=100):
    model.eval()
    dummy = torch.randn(1, 3, 224, 224).half().to(device)
    model = model.half()
    for _ in range(20):
        with torch.inference_mode(): model(dummy)
    if device.type == 'mps': torch.mps.synchronize()
    latencies = []
    for _ in range(n):
        t = time.perf_counter()
        with torch.inference_mode(): model(dummy)
        if device.type == 'mps': torch.mps.synchronize()
        latencies.append((time.perf_counter() - t) * 1000)
    return np.percentile(latencies, 99)

def finetune(model_name, n_epochs=10, lr=2e-5):
    print(f"\n{'='*50}")
    print(f"Fine-tuning {model_name}")
    print(f"{'='*50}")

    # Load pretrained model
    model = timm.create_model(model_name, pretrained=True)
    model = model.to(device)

    # Evaluate before fine-tuning
    acc_before = evaluate(model, val_loader, device)
    print(f"Accuracy before fine-tuning: {acc_before*100:.1f}%")

    # Fine-tune ALL layers -- small/tiny models have fewer params
    # so full fine-tuning is safe and gives better accuracy recovery
    # than attention-only fine-tuning
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    best_acc = acc_before
    best_state = None

    for epoch in range(n_epochs):
        model.train()
        total_loss = 0
        n_batches = 0

        for images, folder_labels in train_loader:
            images = images.to(device)
            true_labels = torch.tensor(
                [folder_to_imagenet[l.item()] for l in folder_labels]
            ).to(device)

            optimizer.zero_grad()
            loss = F.cross_entropy(model(images), true_labels)
            loss.backward()
            # Gradient clipping -- prevents large updates destabilizing
            # the pretrained weights, especially important for tiny models
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        val_acc = evaluate(model, val_loader, device)
        print(f"  Epoch {epoch+1}/{n_epochs} | loss={total_loss/n_batches:.4f} | val_acc={val_acc*100:.1f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            print(f"    ↑ New best: {best_acc*100:.1f}%")

    # Load best checkpoint
    if best_state:
        model.load_state_dict(best_state)

    acc_after = evaluate(model, val_loader, device)
    p99 = benchmark_p99(model, device)

    # Save finetuned model
    save_name = model_name.replace('/', '_')
    torch.save(model.state_dict(), f'checkpoints/{save_name}_finetuned.pt')

    print(f"\n--- {model_name} Results ---")
    print(f"Before fine-tuning: {acc_before*100:.1f}%")
    print(f"After fine-tuning:  {acc_after*100:.1f}%  (+{(acc_after-acc_before)*100:.1f} pts)")
    print(f"P99 latency (FP16): {p99:.1f}ms")
    print(f"Saved: checkpoints/{save_name}_finetuned.pt")

    return acc_before, acc_after, p99

# ── Run fine-tuning for both models ───────────────
acc_before_small, acc_after_small, p99_small = finetune('vit_small_patch16_224', n_epochs=10, lr=2e-5)
acc_before_tiny,  acc_after_tiny,  p99_tiny  = finetune('vit_tiny_patch16_224',  n_epochs=10, lr=2e-5)

# ── Final comparison table ─────────────────────────
print(f"\n{'='*60}")
print(f"FINAL COMPARISON TABLE")
print(f"{'='*60}")
print(f"{'Model':<30} {'P99':>8} {'Before':>8} {'After':>8} {'Gain':>8}")
print(f"{'-'*60}")
print(f"{'ViT-Base FP16 (baseline)':<30} {'24.0ms':>8} {'97.2%':>8} {'97.2%':>8} {'-':>8}")
print(f"{'ViT-Base pruned 25% FP16':<30} {'20.9ms':>8} {'91.9%':>8} {'91.9%':>8} {'-':>8}")
print(f"{'ViT-Small FP16':<30} {p99_small:>7.1f}ms {acc_before_small*100:>7.1f}% {acc_after_small*100:>7.1f}% {(acc_after_small-acc_before_small)*100:>+7.1f}%")
print(f"{'ViT-Tiny FP16':<30} {p99_tiny:>7.1f}ms  {acc_before_tiny*100:>7.1f}% {acc_after_tiny*100:>7.1f}% {(acc_after_tiny-acc_before_tiny)*100:>+7.1f}%")
