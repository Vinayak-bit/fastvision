import torch
import timm
import time
import numpy as np
import pandas as pd
import os

def get_model_size_mb(model):
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.pt') as f:
        torch.save(model.state_dict(), f.name)
        return os.path.getsize(f.name) / 1e6

def benchmark(model, device, n_runs=200):
    model = model.to(device)
    dummy = torch.randn(1, 3, 224, 224).to(device)

    # Warmup runs - not timed, lets the backend "warm up"
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
        'min': np.min(latencies),
        'max': np.max(latencies),
    }

# Setup
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")

model = timm.create_model('vit_base_patch16_224', pretrained=True)
model.eval()

# Benchmark
print("Running benchmark (200 runs)...")
results = benchmark(model, device)
size = get_model_size_mb(model)

print(f"\n=== Baseline Results ({device}) ===")
print(f"P50: {results['p50']:.1f}ms")
print(f"P95: {results['p95']:.1f}ms")
print(f"P99: {results['p99']:.1f}ms")
print(f"Min: {results['min']:.1f}ms")
print(f"Max: {results['max']:.1f}ms")
print(f"Model size: {size:.1f}MB")

# Save checkpoint
torch.save(model.state_dict(), 'checkpoints/baseline.pt')

# Save results to CSV
df = pd.DataFrame([{
    'model': 'baseline',
    'device': str(device),
    'p50_ms': results['p50'],
    'p95_ms': results['p95'],
    'p99_ms': results['p99'],
    'size_mb': size,
}])
df.to_csv('results/results.csv', index=False)
print("\nSaved: checkpoints/baseline.pt, results/results.csv")
