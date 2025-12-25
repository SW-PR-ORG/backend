import os
import pandas as pd
from is_leaked.download_dataset import download_dataset_if_missing

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "rockyou_dataset_cleaned.csv")

# Download once if missing - PASS THE FULL PATH
download_dataset_if_missing(CSV_PATH)

# Load once into memory
df = pd.read_csv(CSV_PATH, usecols=["password"])
password_set = set(df["password"].astype(str))

def is_leaked(password: str) -> bool:
    return password in password_set