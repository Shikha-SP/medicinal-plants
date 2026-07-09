import os
import shutil

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dataset_path = os.path.join(ROOT_DIR, 'data', 'dataset')
for species in os.listdir(dataset_path):
    species_dir = os.path.join(dataset_path, species)
    if len(os.listdir(species_dir)) == 0:
        print(f"Removing empty folder: {species}")
        os.rmdir(species_dir)