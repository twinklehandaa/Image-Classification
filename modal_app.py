import modal
import torch
from torchvision import models, transforms
from PIL import Image
import io
import requests
import os

app = modal.App("catdog-classifier")

HF_MODEL_URL = "https://huggingface.co/twinklehandaa/cat-dog-classifier/resolve/main/catdog_model.pth"
MODEL_LOCAL_PATH = "/tmp/catdog_model.pth"

image = modal.Image.debian_slim().pip_install("torch", "torchvision", "Pillow", "requests")

@app.function(image=image)
def classify_image_bytes(file_bytes: bytes) -> str:
    if not os.path.exists(MODEL_LOCAL_PATH):
        r = requests.get(HF_MODEL_URL)
        with open(MODEL_LOCAL_PATH, "wb") as f:
            f.write(r.content)

    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(MODEL_LOCAL_PATH, map_location="cpu"))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])
    image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(tensor)
        pred = output.argmax(dim=1).item()
        return "Dog" if pred == 1 else "Cat"
