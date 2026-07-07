import os
import json
import shutil
import sys

import faiss
import torch
from PIL import Image
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

index = faiss.read_index("plant_database.index")
MAPPING_FILE = "dataset_image_labels.json"
QUERY_DIR = "queries"
VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".ppm", ".pgm", ".tif", ".tiff", ".webp"}

def is_image_file(filename):
    return os.path.splitext(filename.lower())[1] in VALID_EXTS

def ensure_query_dir():
    os.makedirs(QUERY_DIR, exist_ok=True)

def get_latest_query_image():
    ensure_query_dir()
    candidates = [
        os.path.join(QUERY_DIR, name)
        for name in os.listdir(QUERY_DIR)
        if os.path.isfile(os.path.join(QUERY_DIR, name)) and is_image_file(name)
    ]
    return max(candidates, key=os.path.getmtime) if candidates else None

def copy_to_queries(image_path):
    ensure_query_dir()
    if not os.path.exists(image_path):
        alt_path = os.path.join(QUERY_DIR, image_path)
        if os.path.exists(alt_path):
            image_path = alt_path
        else:
            raise FileNotFoundError(
                f"Query image not found: {image_path}. "
                f"Place the file in the current directory or in '{QUERY_DIR}/'."
            )
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Query path is not a file: {image_path}")
    basename = os.path.basename(image_path)
    destination = os.path.join(QUERY_DIR, basename)
    if os.path.abspath(image_path) != os.path.abspath(destination):
        if os.path.exists(destination):
            name, ext = os.path.splitext(basename)
            counter = 1
            while True:
                destination = os.path.join(QUERY_DIR, f"{name}_{counter}{ext}")
                if not os.path.exists(destination):
                    break
                counter += 1
        shutil.copy2(image_path, destination)
    return destination

def build_image_class_names(dataset_root="dataset", mapping_file=MAPPING_FILE):
    if os.path.exists(mapping_file):
        with open(mapping_file, "r", encoding="utf-8") as f:
            return json.load(f)
    class_names = sorted(
        name for name in os.listdir(dataset_root)
        if os.path.isdir(os.path.join(dataset_root, name))
    )
    image_class_names = []
    for class_name in class_names:
        class_dir = os.path.join(dataset_root, class_name)
        image_files = sorted(
            name for name in os.listdir(class_dir)
            if os.path.isfile(os.path.join(class_dir, name)) and is_image_file(name)
        )
        image_class_names.extend([class_name] * len(image_files))
    with open(mapping_file, "w", encoding="utf-8") as f:
        json.dump(image_class_names, f, ensure_ascii=False)
    return image_class_names

image_class_names = build_image_class_names()

def search_plant(image_path):
    image_path = copy_to_queries(image_path)
    print(f"[search] Loading image: {image_path}")
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)
    print("[search] Extracting features")
    with torch.no_grad():
        vector = model(img_tensor).cpu().numpy().astype("float32")
    vector = vector.reshape(vector.shape[0], -1)
    print("[search] Normalizing query vector")
    faiss.normalize_L2(vector)
    print("[search] Searching database")
    distances, indices = index.search(vector, 3)
    print(f"\n--- Search Results for {image_path} ---")
    for i in range(len(indices[0])):
        idx = indices[0][i]
        plant_name = image_class_names[idx] if idx < len(image_class_names) else f"Unknown(index {idx})"
        print(f"Match {i+1}: {plant_name} (Index {idx}, Distance: {distances[0][i]:.2f})")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        search_plant(sys.argv[1])
    else:
        latest_query = get_latest_query_image()
        if latest_query is None:
            print(f"No query image found in '{QUERY_DIR}'.")
            print("Place an image inside the queries folder or run: python search.py <path_to_image>")
        else:
            print(f"No path provided, using latest query image: {latest_query}")
            search_plant(latest_query)
