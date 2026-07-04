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

def evaluate(model):
    model.eval()
    correct, total = 0, 0
    with torch.inference_mode():
        for images, folder_labels in val_loader:
            images = images.to(device)
            true_labels = torch.tensor(
                [folder_to_imagenet[l.item()] for l in folder_labels]
            ).to(device)
            preds = model(images).argmax(dim=1)
            correct += (preds == true_labels).sum().item()
            total += true_labels.size(0)
    return correct / total

model = timm.create_model('vit_small_patch16_224', pretrained=True)
model = model.to(device)

print(f"Before fine-tuning: {evaluate(model)*100:.1f}%")

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)

best_acc = 0.0

for epoch in range(5):
    model.train()
    total_loss, n_batches = 0, 0
    for images, folder_labels in train_loader:
        images = images.to(device)
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
    val_acc = evaluate(model)
    print(f"Epoch {epoch+1}/5 | loss={total_loss/n_batches:.4f} | val_acc={val_acc*100:.1f}%")

    # Save after EVERY epoch -- never lose progress again
    torch.save(model.state_dict(), f'checkpoints/vit_small_finetuned_epoch{epoch+1}.pt')
    print(f"  Saved: checkpoints/vit_small_finetuned_epoch{epoch+1}.pt")

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), 'checkpoints/vit_small_finetuned_best.pt')
        print(f"  ↑ New best: {best_acc*100:.1f}% → checkpoints/vit_small_finetuned_best.pt")

# Benchmark final model
model = model.half().eval()
dummy = torch.randn(1, 3, 224, 224).half().to(device)
for _ in range(20):
    with torch.inference_mode(): model(dummy)
torch.mps.synchronize()
latencies = []
for _ in range(200):
    t = time.perf_counter()
    with torch.inference_mode(): model(dummy)
    torch.mps.synchronize()
    latencies.append((time.perf_counter() - t) * 1000)

print(f"\n=== ViT-Small Finetuned Final Results ===")
print(f"Accuracy: {best_acc*100:.1f}%")
print(f"P50: {np.percentile(latencies,50):.1f}ms")
print(f"P95: {np.percentile(latencies,95):.1f}ms")
print(f"P99: {np.percentile(latencies,99):.1f}ms")
print(f"\nFull table:")
print(f"  ViT-Base FP16 baseline:      24.0ms  97.2%")
print(f"  ViT-Base pruned 25% FP16:    20.9ms  91.9%")
print(f"  ViT-Small FP16 pretrained:    7.4ms  76.6%")
print(f"  ViT-Small FP16 finetuned:  {np.percentile(latencies,99):.1f}ms  {best_acc*100:.1f}%")
