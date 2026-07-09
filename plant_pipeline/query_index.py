import sys
import os
import json

try:
    import torch
    import numpy as np
    import faiss
    from torchvision import transforms
    from PIL import Image
except ModuleNotFoundError as e:
    print(f"Missing required package: {e.name}.")
    print("Install with pip: pip install torch torchvision faiss-cpu pillow")
    sys.exit(1)


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, 'data')
INDEX_PATH = os.path.join(DATA_DIR, 'plant_database.index')
META_PATH = os.path.join(DATA_DIR, 'plant_database_meta.json')

if not os.path.exists(INDEX_PATH) or not os.path.exists(META_PATH):
    print('Index or metadata not found. Run vectorization.py first.')
    sys.exit(1)

if len(sys.argv) < 2:
    print('Usage: python query_index.py path/to/query.jpg [k]')
    sys.exit(1)

query_path = sys.argv[1]
k = int(sys.argv[2]) if len(sys.argv) > 2 else 5

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
try:
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    model.to(device)
    model.eval()
except Exception:
    from torchvision import models
    resnet = models.resnet50(pretrained=True)
    model = torch.nn.Sequential(*list(resnet.children())[:-1])
    model.to(device)
    model.eval()

img = Image.open(query_path).convert('RGB')
img_t = transform(img).unsqueeze(0).to(device)
with torch.no_grad():
    vec = model(img_t).cpu().numpy()
    if vec.ndim > 2:
        vec = vec.reshape(vec.shape[0], -1)
    vec = vec.astype('float32')

faiss.normalize_L2(vec)

index = faiss.read_index(INDEX_PATH)
with open(META_PATH, 'r', encoding='utf-8') as f:
    meta_list = json.load(f)

D, I = index.search(vec, k)

print('=' * 60)
print(f'Query: {query_path}')
print(f'Top {k} results:')
for dist, idx in zip(D[0], I[0]):
    if idx < 0:
        continue
    meta = meta_list[idx]
    score = 1.0 / (1.0 + float(dist))
    print('-' * 60)
    print(f"Rank score: {score:.3f}   distance: {dist:.6f}   dataset_index: {idx}")
    print(f"Class: {meta.get('class')}")
    print(f"Image: {meta.get('image_path')}")
    csv_meta = meta.get('csv_metadata')
    if csv_meta:
        print('CSV metadata:')
        for kf, vf in csv_meta.items():
            print(f"  {kf}: {vf}")
    else:
        print('No CSV metadata found for this class.')
print('=' * 60)
