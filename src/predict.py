# src/predict.py
# Safe constants for prediction script.

import os
import joblib
import pandas as pd

BASE_DIR = os.path.dirname(__file__)
REPO_ROOT = os.path.normpath(os.path.join(BASE_DIR, ".."))

FEATURES_CSV = os.path.normpath(os.path.join(REPO_ROOT, "data", "processed", "customer_features.csv"))
MODEL_PATH = os.path.normpath(os.path.join(REPO_ROOT, "models", "xgb_cltv_model.joblib"))
REPORTS_DIR = os.path.normpath(os.path.join(REPO_ROOT, "reports"))

def load_features():
    if not os.path.exists(FEATURES_CSV):
        raise FileNotFoundError(f"Features CSV not found at {FEATURES_CSV}")
    return pd.read_csv(FEATURES_CSV)

def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    return joblib.load(MODEL_PATH)
