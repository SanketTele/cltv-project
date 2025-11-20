# src/data_ingest.py
# Safe, relative paths for raw CSV and interim directory.

import os
import pandas as pd

BASE_DIR = os.path.dirname(__file__)  # src/
REPO_ROOT = os.path.normpath(os.path.join(BASE_DIR, ".."))

RAW_CSV_PATH = os.path.normpath(os.path.join(REPO_ROOT, "data", "raw", "online_retail_II.csv"))
INTERIM_DIR = os.path.normpath(os.path.join(REPO_ROOT, "data", "interim"))

def load_raw_transactions():
    if not os.path.exists(RAW_CSV_PATH):
        raise FileNotFoundError(f"Raw CSV not found at {RAW_CSV_PATH}")
    df = pd.read_csv(RAW_CSV_PATH, encoding="latin1", low_memory=False)
    return df

def save_interim(df, filename="transactions_clean.csv"):
    os.makedirs(INTERIM_DIR, exist_ok=True)
    out_path = os.path.join(INTERIM_DIR, filename)
    df.to_csv(out_path, index=False)
    return out_path
