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
META_FILE = "plant_database_meta.json"
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

# load metadata produced by vectorization.py if available
meta_list = None
if os.path.exists(META_FILE):
    try:
        with open(META_FILE, 'r', encoding='utf-8') as f:
            meta_list = json.load(f)
    except Exception as e:
        print(f"Warning: failed to load {META_FILE}: {e}")

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
    # preferred fields to display if present in CSV
    preferred_fields = [
        'scientific_name', 'scientific name', 'class', 'species', 'genus', 'family', 'order', 'phylum', 'kingdom',
        'common_name', 'common name', 'vernacular_name', 'vernacular name', 'description', 'medicinal_uses',
        'area_in_nepal', 'area', 'location', 'habitat', 'id', 'url'
    ]

    def fmt_confidence(dist_val):
        # Map distance to (1 - dist) as requested, then to percentage
        try:
            conf = (1.0 - float(dist_val)) * 100.0
        except Exception:
            conf = 0.0
        if conf < 0:
            conf = 0.0
        if conf > 100:
            conf = 100.0
        return conf

    for i in range(len(indices[0])):
        idx = int(indices[0][i])
        dist = float(distances[0][i])
        csv_meta = None
        display_name = None
        if meta_list and 0 <= idx < len(meta_list):
            meta = meta_list[idx]
            display_name = meta.get('class') or (image_class_names[idx] if idx < len(image_class_names) else None)
            csv_meta = meta.get('csv_metadata')
        else:
            display_name = image_class_names[idx] if idx < len(image_class_names) else f"Unknown(index {idx})"

        conf_pct = fmt_confidence(dist)
        print(f"Match {i+1}: {display_name} (Index {idx}, Distance: {dist:.4f}, Confidence: {conf_pct:.1f}% )")

        if csv_meta:
            # Display the fields that exist in your CSV dataset.
            def print_field(key, label=None):
                value = csv_meta.get(key)
                if value is None or str(value).strip() == "":
                    return
                print(f"  {label or key}: {value}")

            # Primary plant identity fields
            print_field('scientific_name', 'Scientific name')
            print_field('common_name', 'Common name')
            print_field('iconic_taxon_name', 'Iconic taxon')
            print_field('taxon_id', 'Taxon ID')
            print_field('description', 'Description')

            # Nepal location / region fields
            region_parts = []
            for key in ['place_town_name', 'place_county_name', 'place_state_name', 'place_country_name']:
                value = csv_meta.get(key)
                if value and str(value).strip():
                    region_parts.append(str(value).strip())
            if region_parts:
                print(f"  Nepal location: {', '.join(region_parts)}")
            else:
                print_field('place_guess', 'Location guess')

            # Taxonomy hierarchy
            taxon_keys = [
                ('taxon_kingdom_name', 'Kingdom'),
                ('taxon_phylum_name', 'Phylum'),
                ('taxon_subphylum_name', 'Subphylum'),
                ('taxon_superclass_name', 'Superclass'),
                ('taxon_class_name', 'Class'),
                ('taxon_subclass_name', 'Subclass'),
                ('taxon_superorder_name', 'Superorder'),
                ('taxon_order_name', 'Order'),
                ('taxon_suborder_name', 'Suborder'),
                ('taxon_superfamily_name', 'Superfamily'),
                ('taxon_family_name', 'Family'),
                ('taxon_subfamily_name', 'Subfamily'),
                ('taxon_supertribe_name', 'Supertribe'),
                ('taxon_tribe_name', 'Tribe'),
                ('taxon_subtribe_name', 'Subtribe'),
                ('taxon_genus_name', 'Genus'),
                ('taxon_species_name', 'Species'),
                ('taxon_subspecies_name', 'Subspecies'),
                ('taxon_variety_name', 'Variety'),
                ('taxon_form_name', 'Form'),
            ]
            printed_taxon = False
            for key, label in taxon_keys:
                if csv_meta.get(key) and str(csv_meta.get(key)).strip():
                    if not printed_taxon:
                        print("  Taxonomy:")
                        printed_taxon = True
                    print(f"    {label}: {csv_meta.get(key)}")

            # Additional useful fields if present
            print_field('id', 'Observation ID')
            print_field('url', 'Observation URL')
            print_field('image_url', 'Image URL')
            print_field('time_zone', 'Time zone')
            print_field('quality_grade', 'Quality grade')
            print_field('captive_cultivated', 'Captive/cultivated')
        else:
            print("  No CSV metadata available for this match.")

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
