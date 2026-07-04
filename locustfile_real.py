from locust import HttpUser, task, between
import os, random

IMAGE_DIRS = [
    f"data/imagenette2-320/val/{cls}"
    for cls in os.listdir("data/imagenette2-320/val")
    if not cls.startswith(".")
]
IMAGES = []
for d in IMAGE_DIRS:
    for f in os.listdir(d)[:5]:
        if f.endswith('.JPEG') or f.endswith('.jpg'):
            IMAGES.append(os.path.join(d, f))

print(f"Loaded {len(IMAGES)} images across {len(IMAGE_DIRS)} classes")

class VisionUser(HttpUser):
    wait_time = between(0.0, 0.1)

    @task
    def predict(self):
        path = random.choice(IMAGES)
        with open(path, 'rb') as f:
            self.client.post("/predict", files={"file": ("image.jpg", f, "image/jpeg")})
