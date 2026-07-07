import sys
import os
try:
    import torch
    import numpy as np
    import faiss
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader
except ModuleNotFoundError as e:
    print(f"Missing required package: {e.name}.")
    print("Install with pip (cpu): pip install torch torchvision faiss-cpu")
    print("Or use conda for preferred CUDA builds: conda install -c pytorch pytorch torchvision faiss-gpu -c pytorch")
    sys.exit(1)

if not os.path.exists('dataset'):
    print("Dataset folder 'dataset' not found. Please create it with subfolders per class.")
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

dataset = datasets.ImageFolder("dataset", transform=transform)
if len(dataset) == 0:
    print("No images found in 'dataset'.")
    sys.exit(1)

dataloader = DataLoader(dataset, batch_size=64, shuffle=False)

print(f"[vectorization] Found {len(dataset)} images across {len(dataset.classes)} classes")
print(f"[vectorization] Batch size={dataloader.batch_size}, total batches={len(dataloader)}")

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
faiss.write_index(index, "plant_database.index")
print("[vectorization] Index saved to plant_database.index")
print("[vectorization] Done.")