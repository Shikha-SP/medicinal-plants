import sys
import threading
try:
    import pandas as pd
    import os
    import requests
    from concurrent.futures import ThreadPoolExecutor
except ModuleNotFoundError as e:
    print(f"Missing required package: {e.name}. Install dependencies with: pip install -r requirements.txt")
    sys.exit(1)

counter = 0
counter_lock = threading.Lock()

def get_existing_file_count():
    count = 0
    if os.path.exists('dataset'):
        for root, dirs, files in os.walk('dataset'):
            count += len([f for f in files if f.endswith('.jpg')])
    return count

def download_single_image(row, total_rows):
    global counter
    
    species = str(row['scientific_name']).replace(" ", "_")
    url = row['image_url']
    if pd.isna(url): return None
    
    species_dir = os.path.join('dataset', species)
    os.makedirs(species_dir, exist_ok=True)
    file_path = os.path.join(species_dir, f"{row['id']}.jpg")
    
    if not os.path.exists(file_path):
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                with open(file_path, 'wb') as f:
                    f.write(response.content)
                with counter_lock:
                    counter += 1
                    print(f"[{counter + initial_count}/{total_rows}] Saved: {row['id']}.jpg into {species}")
        except Exception as e:
            print(f"Failed to download {url}: {e}")
            return None
    return None

def download_images(csv_path):
    global initial_count
    df = pd.read_csv(csv_path, encoding='utf-8-sig', low_memory=False)
    total_rows = len(df)
    
    initial_count = get_existing_file_count()
    print(f"Starting from {initial_count} existing files...")
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        executor.map(lambda row: download_single_image(row, total_rows), [row for _, row in df.iterrows()])
    
    print("Download process complete.")

if __name__ == "__main__":
    download_images('raw_observations.csv')