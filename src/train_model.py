# src/train_model.py
# Training script constants using safe paths.

import os
import joblib
import pandas as pd

BASE_DIR = os.path.dirname(__file__)
REPO_ROOT = os.path.normpath(os.path.join(BASE_DIR, ".."))

FEATURES_CSV = os.path.normpath(os.path.join(REPO_ROOT, "data", "processed", "customer_features.csv"))
MODEL_DIR = os.path.normpath(os.path.join(REPO_ROOT, "models"))
REPORTS_DIR = os.path.normpath(os.path.join(REPO_ROOT, "reports"))
MODEL_PATH = os.path.join(MODEL_DIR, "xgb_cltv_model.joblib")

def ensure_dirs():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

def save_model(model):
    ensure_dirs()
    joblib.dump(model, MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")
    return MODEL_PATH
