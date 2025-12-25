import os
import zipfile
import requests

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PCFG_BASE_DIR = os.path.join(BASE_DIR, "pcfg_cracker")
RULES_DIR = os.path.join(PCFG_BASE_DIR, "Rules")

# Google Drive FILE ID (from your link)
GOOGLE_DRIVE_FILE_ID = "1Bw_cYz5DF1XyjmEDB9OWj_7qFAsUEcWG"

ZIP_NAME = "pcfg_rules.zip"
ZIP_PATH = os.path.join(PCFG_BASE_DIR, ZIP_NAME)

DOWNLOAD_URL = f"https://drive.google.com/uc?export=download&id={GOOGLE_DRIVE_FILE_ID}"


def download_and_extract_rules():
    # Rules already present → skip
    if os.path.exists(RULES_DIR) and os.listdir(RULES_DIR):
        print("PCFG Rules already exist — skipping download")
        return

    print("Downloading PCFG Rules ZIP from Google Drive...")
    os.makedirs(PCFG_BASE_DIR, exist_ok=True)

    # Download ZIP
    response = requests.get(DOWNLOAD_URL, stream=True)
    response.raise_for_status()

    with open(ZIP_PATH, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    print("Download complete. Extracting...")

    # Extract ZIP
    with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall(PCFG_BASE_DIR)

    # Remove ZIP after extraction
    os.remove(ZIP_PATH)

    print("PCFG Rules extracted successfully.")
