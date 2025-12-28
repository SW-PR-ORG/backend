import os
import gdown
import zipfile

def download_and_extract_pcfg():
    
    # URL of the shared file (make sure it's the "shareable link")
    gdrive_url = "https://drive.google.com/uc?id=1Bw_cYz5DF1XyjmEDB9OWj_7qFAsUEcWG&export=download"
    
    # Paths
    base_dir = os.path.dirname(os.path.abspath(__file__))  # features folder
    target_dir = os.path.join(base_dir, "pcfg_cracker")
    zip_path = os.path.join(target_dir, "pcfg_rules.zip")

    # Make sure target directory exists
    os.makedirs(target_dir, exist_ok=True)

    # Download the file
    print("Downloading PCFG zip from Google Drive...")
    gdown.download(gdrive_url, zip_path, quiet=False)

    # Extract zip
    print("Extracting PCFG zip...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(target_dir)

    # Delete zip
    print("Deleting zip file...")
    os.remove(zip_path)

    print("PCFG rules ready!")

