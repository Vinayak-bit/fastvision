import torch
import timm
import time
import numpy as np
import pandas as pd
import copy
import os

def get_model_size_mb(model):
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.pt') as f:
        torch.save(model.state_dict(), f.name)
        return os.path.getsize(f.name) / 1e6

def benchmark(model, device, n_runs=200, dtype=torch.float32):
    model = model.to(device)
    dummy = torch.randn(1, 3, 224, 224, dtype=dtype).to(device)

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

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")

# Load baseline model
model = timm.create_model('vit_base_patch16_224', pretrained=True)
model.eval()

# ── FP16 on MPS ───────────────────────────────────
print("\nBenchmarking FP16 on MPS...")
model_fp16 = copy.deepcopy(model).to(device).half()
fp16_results = benchmark(model_fp16, device, dtype=torch.float16)
fp16_size = get_model_size_mb(model_fp16)

print(f"FP16 MPS P50: {fp16_results['p50']:.1f}ms")
print(f"FP16 MPS P95: {fp16_results['p95']:.1f}ms")
print(f"FP16 MPS P99: {fp16_results['p99']:.1f}ms")
print(f"FP16 size: {fp16_size:.1f}MB")

torch.save(model_fp16.state_dict(), 'checkpoints/fp16.pt')

# ── INT8 PTQ on CPU (MPS doesn't support INT8 quantization) ──
print("\nBenchmarking INT8 PTQ on CPU (MPS limitation - this is expected)...")
model_cpu = copy.deepcopy(model).to('cpu')
model_cpu.eval()

try:
    model_cpu.qconfig = torch.quantization.get_default_qconfig('qnnpack')
    torch.quantization.prepare(model_cpu, inplace=True)

    # Calibrate with a few dummy passes
    with torch.no_grad():
        for _ in range(5):
            dummy_cpu = torch.randn(1, 3, 224, 224)
            model_cpu(dummy_cpu)

    torch.quantization.convert(model_cpu, inplace=True)

    int8_results = benchmark(model_cpu, torch.device('cpu'))
    int8_size = get_model_size_mb(model_cpu)

    print(f"INT8 CPU P50: {int8_results['p50']:.1f}ms")
    print(f"INT8 CPU P95: {int8_results['p95']:.1f}ms")
    print(f"INT8 CPU P99: {int8_results['p99']:.1f}ms")
    print(f"INT8 size: {int8_size:.1f}MB")

    torch.save(model_cpu.state_dict(), 'checkpoints/int8_cpu.pt')
except Exception as e:
    print(f"INT8 quantization failed/unsupported: {e}")
    int8_results = None
    int8_size = None

# ── Save comparison ───────────────────────────────
print("\n=== Summary ===")
baseline_df = pd.read_csv('results/results.csv')
print(baseline_df.to_string(index=False))

new_rows = [{
    'model': 'fp16_mps',
    'device': str(device),
    'p50_ms': fp16_results['p50'],
    'p95_ms': fp16_results['p95'],
    'p99_ms': fp16_results['p99'],
    'size_mb': fp16_size,
}]

if int8_results:
    new_rows.append({
        'model': 'int8_ptq_cpu',
        'device': 'cpu',
        'p50_ms': int8_results['p50'],
        'p95_ms': int8_results['p95'],
        'p99_ms': int8_results['p99'],
        'size_mb': int8_size,
    })

df = pd.concat([baseline_df, pd.DataFrame(new_rows)], ignore_index=True)
df.to_csv('results/results.csv', index=False)
print("\n", df.to_string(index=False))
print("\nSaved: checkpoints/fp16.pt, results/results.csv updated")
