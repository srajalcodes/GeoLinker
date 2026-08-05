import os
import urllib.request
import zipfile
from pathlib import Path

def download_file(url, dest_path):
    if not dest_path.exists():
        print(f"Downloading {dest_path.name} (this might take a few minutes)...")
        # Added headers to prevent Zenodo from blocking the automated request
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response, open(dest_path, 'wb') as out_file:
            out_file.write(response.read())
        print("Done!")
    else:
        print(f"File {dest_path.name} already exists. Skipping.")

def main():
    # 1. Create directories
    dirs = ['datasets', 'checkpoints']
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)

    # 2. Download Checkpoint from your Zenodo
    print("\n--- Downloading Checkpoint ---")
    checkpoint_url = "https://zenodo.org/records/21806622/files/geolinker_best_unified.pt?download=1"
    download_file(checkpoint_url, Path('checkpoints/geolinker_best_unified.pt'))

    # 3. Download Datasets ZIP from your Zenodo
    print("\n--- Downloading Datasets ---")
    datasets_zip_url = "https://zenodo.org/records/21806622/files/geolinker_datasets.zip?download=1"
    zip_path = Path('datasets/geolinker_datasets.zip')
    download_file(datasets_zip_url, zip_path)

    # 4. Extract Datasets
    print("\n--- Extracting Datasets ---")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall('datasets/')
    print("Extraction complete!")
    
    # Clean up the zip file to save space
    os.remove(zip_path)

    print("\n✅ All data and checkpoints downloaded and extracted successfully. You are ready to run reproduce.py!")

if __name__ == "__main__":
    main()