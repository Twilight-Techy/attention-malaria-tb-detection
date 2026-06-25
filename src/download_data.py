import os
import subprocess
from pathlib import Path

def setup_kaggle_credentials():
    # Kaggle API will automatically use KAGGLE_USERNAME and KAGGLE_KEY environment variables if set.
    # Otherwise, it looks for ~/.kaggle/kaggle.json
    pass

def download_dataset(dataset_name, download_path, check_folder):
    print(f"Checking {dataset_name}...")
    if os.path.exists(os.path.join(download_path, check_folder)):
        print(f"-> {dataset_name} already exists at {download_path}/{check_folder}. Skipping download.")
        return

    print(f"-> Downloading {dataset_name}...")
    try:
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", dataset_name, "-p", download_path, "--unzip"],
            check=True
        )
        print(f"Successfully downloaded and unzipped {dataset_name} to {download_path}")
    except Exception as e:
        print(f"Failed to download {dataset_name}: {e}")
        print("Please ensure kaggle.json is configured correctly.")

if __name__ == "__main__":
    setup_kaggle_credentials()
    
    base_dir = Path(os.path.dirname(os.path.abspath(__file__))).parent
    data_dir = base_dir / "data"
    data_dir.mkdir(exist_ok=True)
    
    # Malaria dataset: NIH Cell Images
    malaria_dataset = "iarunava/cell-images-for-detecting-malaria"
    malaria_path = data_dir / "malaria"
    malaria_path.mkdir(exist_ok=True)
    
    # TB Dataset: Shenzhen and Montgomery
    # Using a widely used combined dataset that matches the report's requirements
    tb_dataset = "tawsifurrahman/tuberculosis-tb-chest-xray-dataset"
    tb_path = data_dir / "tuberculosis"
    tb_path.mkdir(exist_ok=True)
    
    download_dataset(malaria_dataset, str(malaria_path), "cell_images")
    download_dataset(tb_dataset, str(tb_path), "TB_Chest_Radiography_Database")
    
    print("\nData acquisition phase complete!")
