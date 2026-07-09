import io
import os
import json
import shutil
import sys
import warnings
from contextlib import redirect_stderr, redirect_stdout

import faiss
import torch
from PIL import Image
from torchvision import transforms

warnings.filterwarnings("ignore", message="xFormers is not available.*")

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")
INDEX_PATH = os.path.join(DATA_DIR, "plant_database.index")
MAPPING_FILE = os.path.join(DATA_DIR, "dataset_image_labels.json")
META_FILE = os.path.join(DATA_DIR, "plant_database_meta.json")
QUERY_DIR = os.path.join(DATA_DIR, "queries")
VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".ppm", ".pgm", ".tif", ".tiff", ".webp"}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

index = faiss.read_index(INDEX_PATH)


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


def build_image_class_names(dataset_root=os.path.join(DATA_DIR, "dataset"), mapping_file=MAPPING_FILE):
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

meta_list = None
if os.path.exists(META_FILE):
    try:
        with open(META_FILE, 'r', encoding='utf-8') as f:
            meta_list = json.load(f)
    except Exception as e:
        print(f"Warning: failed to load {META_FILE}: {e}")


def fmt_confidence(dist_val):
    try:
        conf = max(0.0, min(100.0, (1.0 - float(dist_val)) * 100.0))
    except Exception:
        conf = 0.0
    return conf


def fmt_confidence_label(conf_pct):
    if conf_pct >= 90:
        return "very high"
    if conf_pct >= 75:
        return "high"
    if conf_pct >= 55:
        return "moderate"
    if conf_pct >= 35:
        return "low"
    return "very low"


def get_first_value(meta, keys):
    if not meta:
        return None
    for key in keys:
        value = meta.get(key)
        if value is not None and str(value).strip() != "":
            return str(value).strip()
    return None


def summarize_metadata(meta):
    if not meta:
        return {}

    details = {}

    scientific_name = get_first_value(meta, ['scientific_name', 'scientific name'])
    common_name = get_first_value(meta, ['common_name', 'common name'])
    description = get_first_value(meta, ['description'])
    observation_url = get_first_value(meta, ['url'])
    image_url = get_first_value(meta, ['image_url'])

    if scientific_name:
        details['scientific_name'] = scientific_name
    if common_name:
        details['common_name'] = common_name
    if description:
        details['description'] = description
    if observation_url:
        details['observation_url'] = observation_url
    if image_url:
        details['image_url'] = image_url

    taxonomy = []
    taxon_keys = [
        ('taxon_kingdom_name', 'Kingdom'),
        ('taxon_phylum_name', 'Phylum'),
        ('taxon_class_name', 'Class'),
        ('taxon_order_name', 'Order'),
        ('taxon_family_name', 'Family'),
        ('taxon_genus_name', 'Genus'),
        ('taxon_species_name', 'Species'),
    ]
    for key, label in taxon_keys:
        value = meta.get(key)
        if value and str(value).strip():
            taxonomy.append({'label': label, 'value': str(value).strip()})
    if taxonomy:
        details['taxonomy'] = taxonomy

    return details


def search_plant(image_path, verbose=True):
    image_path = copy_to_queries(image_path)

    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        vector = model(img_tensor).cpu().numpy().astype("float32")
    vector = vector.reshape(vector.shape[0], -1)
    faiss.normalize_L2(vector)
    distances, indices = index.search(vector, 3)

    matches = []
    for i in range(len(indices[0])):
        idx = int(indices[0][i])
        dist = float(distances[0][i])
        meta = None
        display_name = None
        if meta_list and 0 <= idx < len(meta_list):
            meta = meta_list[idx]
            display_name = meta.get('class') or (
                image_class_names[idx] if idx < len(image_class_names) else None
            )
        else:
            display_name = (
                image_class_names[idx] if idx < len(image_class_names) else f"Unknown (index {idx})"
            )

        conf_pct = fmt_confidence(dist)
        confidence_label = fmt_confidence_label(conf_pct)
        match_payload = {
            'rank': i + 1,
            'name': display_name,
            'confidence': round(conf_pct, 1),
            'confidence_label': confidence_label,
            'index': idx,
            'distance': round(dist, 6),
            'metadata': summarize_metadata(meta.get('csv_metadata') if meta else None),
        }
        matches.append(match_payload)

    result = {
        'image_path': image_path,
        'best_match': matches[0],
        'other_matches': matches[1:],
    }

    if verbose:
        best_match = result['best_match']
        print("")
        print("")
        print(f"Best match: {best_match['name']} ({best_match['confidence']:.1f}% confidence, {best_match['confidence_label']})")
        print("")

        metadata = best_match.get('metadata', {})
        if metadata:
            if metadata.get('scientific_name'):
                print(f"Scientific name: {metadata['scientific_name']}")
            if metadata.get('common_name'):
                print(f"Common name: {metadata['common_name']}")
            if metadata.get('description'):
                print(f"Notes: {metadata['description']}")
            if metadata.get('taxonomy'):
                taxonomy_text = ' > '.join(f"{item['label']}: {item['value']}" for item in metadata['taxonomy'])
                print()
                print(f"Taxonomy: {taxonomy_text}")
            if metadata.get('observation_url'):
                print()
                print(f"Observation: {metadata['observation_url']}")
            if metadata.get('image_url'):
                print(f"Image: {metadata['image_url']}")
        else:
            print("No additional metadata available for this match.")

        if len(matches) > 1:
            print()
            print("Other possible matches:")
            for match in matches[1:]:
                print(f"  - {match['name']} ({match['confidence']:.1f}% confidence, {match['confidence_label']})")

    return result


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
    print("")
    print("")