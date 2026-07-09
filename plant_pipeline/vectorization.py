
import sys
import os
import csv
import json

try:
    import torch
    import numpy as np
    import faiss
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader
    from PIL import Image
except ModuleNotFoundError as e:
    print(f"Missing required package: {e.name}.")
    print("Install with pip (cpu): pip install torch torchvision faiss-cpu pillow")
    print("Or use conda for preferred CUDA builds: conda install -c pytorch pytorch torchvision faiss-gpu -c pytorch")
    sys.exit(1)


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")
DATASET_DIR = os.path.join(DATA_DIR, "dataset")
CSV_PATH = os.path.join(DATA_DIR, "raw_observations.csv")
INDEX_PATH = os.path.join(DATA_DIR, "plant_database.index")
META_PATH = os.path.join(DATA_DIR, "plant_database_meta.json")

if not os.path.exists(DATASET_DIR):
    print("Dataset folder 'data/dataset' not found. Please create it with subfolders per class.")
    sys.exit(1)


print("[vectorization] Starting")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[vectorization] Using device: {device}")
try:
    print("[vectorization] Loading DINOv2 model...")
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    model.to(device)
    model.eval()
except Exception as e:
    print("Failed to load DINOv2 model:", e)
    print("Falling back to ResNet50.")
    try:
        from torchvision import models
        resnet = models.resnet50(pretrained=True)
        model = torch.nn.Sequential(*list(resnet.children())[:-1])
        model.to(device)
        model.eval()
        print("Fallback ResNet50 loaded.")
    except Exception as e2:
        print("Fallback failed:", e2)
        sys.exit(1)


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


dataset = datasets.ImageFolder(DATASET_DIR, transform=transform)
if len(dataset) == 0:
    print("No images found in 'dataset'.")
    sys.exit(1)


dataloader = DataLoader(dataset, batch_size=64, shuffle=False)


print(f"[vectorization] Found {len(dataset)} images across {len(dataset.classes)} classes")
print(f"[vectorization] Batch size={dataloader.batch_size}, total batches={len(dataloader)}")


meta_list = []

csv_rows = []
csv_path = CSV_PATH
if os.path.exists(csv_path):
    try:
        with open(csv_path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for r in reader:
                csv_rows.append(r)
    except Exception as e:
        print(f"Warning: failed to read {csv_path}:", e)


def _norm(s: str) -> str:
    return s.strip().lower().replace('_', ' ')

lookup_dict = {}
for r in csv_rows:
    for v in r.values():
        if not v:
            continue
        key = _norm(str(v))
        if key not in lookup_dict:
            lookup_dict[key] = r


def find_csv_for_class(class_name):
    key = _norm(class_name)
    if key in lookup_dict:
        return lookup_dict[key]
    for r in csv_rows:
        for v in r.values():
            if not v:
                continue
            if key in _norm(str(v)):
                return r
    return None


for img_path, class_idx in dataset.samples:
    class_name = dataset.classes[class_idx]
    meta = find_csv_for_class(class_name) if csv_rows else None
    meta_list.append({
        'image_path': img_path,
        'class': class_name,
        'csv_metadata': meta,
    })


all_vectors = []
with torch.no_grad():
    processed_batches = 0
    total_batches = len(dataloader)
    for images, _ in dataloader:
        images = images.to(device)
        try:
            vectors = model(images)
        except Exception as e:
            print("[vectorization] Model forward pass failed:", e)
            sys.exit(1)
        arr = vectors.cpu().numpy()
        if arr.ndim > 2:
            arr = arr.reshape(arr.shape[0], -1)
        all_vectors.append(arr)
        processed_batches += 1
        if processed_batches % 10 == 0 or processed_batches == total_batches:
            print(f"[vectorization] Processed {processed_batches}/{total_batches} batches; last batch shape: {arr.shape}")


if not all_vectors:
    print("No vectors computed.")
    sys.exit(1)


all_vectors = np.vstack(all_vectors).astype('float32')
print(f"[vectorization] Stacked vectors shape: {all_vectors.shape}")
print("[vectorization] Normalizing vectors to unit length")

faiss.normalize_L2(all_vectors)

index = faiss.IndexFlatL2(all_vectors.shape[1])
print(f"[vectorization] Adding {all_vectors.shape[0]} vectors to index")
index.add(all_vectors)
faiss.write_index(index, INDEX_PATH)
print(f"[vectorization] Index saved to {INDEX_PATH}")

try:
    with open(META_PATH, 'w', encoding='utf-8') as f:
        json.dump(meta_list, f, ensure_ascii=False, indent=2)
    print(f"[vectorization] Saved metadata to {META_PATH}")
except Exception as e:
    print("[vectorization] Failed to save metadata:", e)

print("[vectorization] Done.")
