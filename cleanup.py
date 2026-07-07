import os
import shutil

dataset_path = 'dataset'
for species in os.listdir(dataset_path):
    species_dir = os.path.join(dataset_path, species)
    # If the folder is empty, remove it
    if len(os.listdir(species_dir)) == 0:
        print(f"Removing empty folder: {species}")
        os.rmdir(species_dir)