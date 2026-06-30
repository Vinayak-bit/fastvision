import torch
import timm
import json
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

IMAGENETTE_WNID_TO_IMAGENET_IDX = {
    'n01440764': 0, 'n02102040': 217, 'n02979186': 482, 'n03000684': 491,
    'n03028079': 497, 'n03394916': 566, 'n03417042': 569, 'n03425413': 571,
    'n03445777': 574, 'n03888257': 701,
}

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_dataset = datasets.ImageFolder('data/imagenette2-320/val', transform=transform)
print(f"Found {len(val_dataset)} validation images across {len(val_dataset.classes)} classes")

folder_idx_to_imagenet_idx = {
    i: IMAGENETTE_WNID_TO_IMAGENET_IDX[wnid]
    for i, wnid in enumerate(val_dataset.classes)
}

val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)


def evaluate(model, loader, device, folder_to_imagenet, max_batches=None):
    model = model.to(device)
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for i, (images, folder_labels) in enumerate(loader):
            if max_batches and i >= max_batches:
                break
            images = images.to(device)
            imagenet_labels = torch.tensor(
                [folder_to_imagenet[l.item()] for l in folder_labels]
            ).to(device)
            out = model(images)
            preds = out.argmax(dim=1)
            correct += (preds == imagenet_labels).sum().item()
            total += imagenet_labels.size(0)
    return correct / total


print("\n=== Evaluating Baseline (FP32) ===")
baseline_model = timm.create_model('vit_base_patch16_224', pretrained=True)
baseline_model.eval()
baseline_acc = evaluate(baseline_model, val_loader, device, folder_idx_to_imagenet_idx, max_batches=20)
print(f"Baseline accuracy: {baseline_acc:.3f} ({baseline_acc*100:.1f}%)")

print("\n=== Evaluating Pruned 50% (real-data importance) ===")

# IMPORTANT: load the importance scores that were ACTUALLY used to prune the
# current checkpoint, not a stale copy -- this file was just overwritten by pruning.py
with open('results/head_importance.json') as f:
    importance = json.load(f)
importance = {int(k): v for k, v in importance.items()}

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

heads_to_prune = get_heads_to_prune(importance, ratio=0.25)
print(f"Heads pruned by layer (must match pruning.py output): {heads_to_prune}")

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
    new_qkv = torch.nn.Linear(embed_dim, len(keep_indices), bias=attn.qkv.bias is not None)
    attn.qkv = new_qkv
    keep_head_cols = []
    for h in keep_heads:
        start = h * head_dim
        keep_head_cols.extend(range(start, start + head_dim))
    new_proj = torch.nn.Linear(len(keep_head_cols), embed_dim, bias=attn.proj.bias is not None)
    attn.proj = new_proj
    attn.num_heads = new_num_heads
    attn.attn_dim = new_num_heads * head_dim

fresh_model = timm.create_model('vit_base_patch16_224', pretrained=False)
for layer_idx, heads in heads_to_prune.items():
    block = fresh_model.blocks[layer_idx]
    prune_attention_heads(block, set(heads))

result = fresh_model.load_state_dict(torch.load('checkpoints/pruned_50.pt', map_location='cpu'))
print(f"load_state_dict -- missing: {result.missing_keys}, unexpected: {result.unexpected_keys}")
fresh_model.eval()

pruned_acc = evaluate(fresh_model, val_loader, device, folder_idx_to_imagenet_idx, max_batches=20)
print(f"Pruned 50% accuracy: {pruned_acc:.3f} ({pruned_acc*100:.1f}%)")

print(f"\n=== Accuracy Summary ===")
print(f"Baseline:   {baseline_acc*100:.1f}%")
print(f"Pruned 50%: {pruned_acc*100:.1f}%")
print(f"Drop:       {(baseline_acc - pruned_acc)*100:.1f} percentage points")
