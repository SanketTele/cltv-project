# src/features.py
# Safe path constants for features pipeline.

import os
import pandas as pd

BASE_DIR = os.path.dirname(__file__)
REPO_ROOT = os.path.normpath(os.path.join(BASE_DIR, ".."))

INTERIM_CSV = os.path.normpath(os.path.join(REPO_ROOT, "data", "interim", "transactions_clean.csv"))
PROCESSED_DIR = os.path.normpath(os.path.join(REPO_ROOT, "data", "processed"))
REPORTS_DIR = os.path.normpath(os.path.join(REPO_ROOT, "reports"))

def load_interim():
    if not os.path.exists(INTERIM_CSV):
        raise FileNotFoundError(f"Interim CSV not found at {INTERIM_CSV}")
    return pd.read_csv(INTERIM_CSV)

def save_processed(df, filename="customer_features.csv"):
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    out_path = os.path.join(PROCESSED_DIR, filename)
    df.to_csv(out_path, index=False)
    return out_path
