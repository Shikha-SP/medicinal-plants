"""
predict.py — Plant identification using Sikha's FAISS index + DINOv2
Drop this file in your project root and import in main.py
"""

import os
import sys
import json
import numpy as np
from PIL import Image
from io import BytesIO

# Add plant_pipeline to path so we can use Sikha's code
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'plant_pipeline'))

# Paths
INDEX_PATH = os.path.join(ROOT_DIR, 'data', 'plant_database.index')
META_PATH = os.path.join(ROOT_DIR, 'data', 'plant_database_meta.json')
LABELS_PATH = os.path.join(ROOT_DIR, 'data', 'dataset_image_labels.json')

try:
    import torch
    import faiss
    from torchvision import transforms
except ModuleNotFoundError as e:
    print(f"Missing package: {e.name}")
    print("Install with: pip install torch torchvision faiss-cpu")
    raise

# ── Load FAISS index and metadata once at startup ────────
print("[predict] Loading FAISS index...")
if not os.path.exists(INDEX_PATH):
    raise FileNotFoundError(f"plant_database.index not found at {INDEX_PATH}")

faiss_index = faiss.read_index(INDEX_PATH)

print("[predict] Loading plant metadata...")
with open(META_PATH, 'r', encoding='utf-8') as f:
    meta_list = json.load(f)

print(f"[predict] FAISS index loaded — {faiss_index.ntotal} vectors")

# ── Load DINOv2 model ─────────────────────────────────────
print("[predict] Loading DINOv2 model...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

try:
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    model.to(device)
    model.eval()
    print("[predict] DINOv2 loaded successfully")
except Exception:
    print("[predict] DINOv2 failed, falling back to ResNet50...")
    from torchvision import models
    resnet = models.resnet50(pretrained=True)
    model = torch.nn.Sequential(*list(resnet.children())[:-1])
    model.to(device)
    model.eval()
    print("[predict] ResNet50 loaded as fallback")

# ── Image transform ───────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ── Main prediction function ──────────────────────────────
def predict_plant(image: Image.Image, k: int = 3) -> dict:
    """
    Takes a PIL Image, returns plant identification result.
    
    Returns:
    {
        "plant_name": "Ocimum_tenuiflorum",
        "confidence": 0.759,
        "top_matches": [
            {"plant_name": "Ocimum_tenuiflorum", "confidence": 0.759},
            {"plant_name": "Ocimum_basilicum", "confidence": 0.634},
        ]
    }
    """
    # Preprocess image
    img_tensor = transform(image).unsqueeze(0).to(device)

    # Get embedding from model
    with torch.no_grad():
        vec = model(img_tensor).cpu().numpy()
        if vec.ndim > 2:
            vec = vec.reshape(vec.shape[0], -1)
        vec = vec.astype('float32')

    # Normalize for cosine similarity
    faiss.normalize_L2(vec)

    # Search FAISS index
    D, I = faiss_index.search(vec, k)

    # Build results
    top_matches = []
    for dist, idx in zip(D[0], I[0]):
        if idx < 0 or idx >= len(meta_list):
            continue
        meta = meta_list[idx]
        score = float(1.0 / (1.0 + float(dist)))
        plant_class = meta.get('class', 'Unknown')
        top_matches.append({
            "plant_name": plant_class,
            "confidence": round(score, 3)
        })

    if not top_matches:
        return {
            "plant_name": "Unknown",
            "confidence": 0.0,
            "top_matches": []
        }

    best = top_matches[0]
    return {
        "plant_name": best["plant_name"],
        "confidence": best["confidence"],
        "top_matches": top_matches
    }
