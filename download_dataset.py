import os
import gdown

FILE_ID = "1j_TJtld3_-tFwL4Kt1i6APqDSObSLMlQ"
FILENAME = "rockyou_dataset_cleaned.csv"

def download_dataset_if_missing(file_path: str):
    if os.path.exists(file_path):
        return

    print("Downloading leaked password dataset...")
    url = f"https://drive.google.com/uc?id={FILE_ID}"
    gdown.download(url, file_path, quiet=False)