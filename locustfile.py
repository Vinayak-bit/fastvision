from locust import HttpUser, task, between
import torch
import numpy as np

# Pre-generate random tensors once at module load --
# all Locust workers share this, so no per-request tensor
# generation cost. Simulates preprocessed image inputs.
TENSORS = [
    torch.randn(1, 3, 224, 224).half().numpy().tobytes()
    for _ in range(20)
]

import random

class VisionUser(HttpUser):
    wait_time = between(0.0, 0.05)  # 0-50ms between requests

    @task
    def predict(self):
        payload = random.choice(TENSORS)
        self.client.post(
            "/predict_tensor",
            data=payload,
            headers={"Content-Type": "application/octet-stream"}
        )
