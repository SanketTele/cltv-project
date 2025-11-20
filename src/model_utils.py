# src/model_utils.py
# Safe path handling, no raw Windows paths.

import os
import joblib
import pandas as pd
from typing import List, Tuple, Any

def load_model_and_feature_order():
    base_dir = os.path.dirname(__file__)  # src/
    model_path = os.path.normpath(os.path.join(base_dir, "..", "models", "xgb_cltv_model.joblib"))

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    model = joblib.load(model_path)

    feature_order: List[str] = [
        "frequency",
        "total_spend",
        "aov",
        "recency_days",
        "T_days",
        "avg_interpurchase_days",
        "active_months",
        "purchase_days_std",
        "category_diversity",
        "avg_order_value",
        "unique_days"
    ]

    return model, feature_order

def prepare_input_df(customers: List[dict], feature_order: List[str]) -> Tuple[Any, List[str]]:
    df = pd.DataFrame(customers)

    required_cols = ["customer_id"] + feature_order
    for col in required_cols:
        if col not in df.columns:
            df[col] = 0.0

    df = df[["customer_id"] + feature_order].copy()
    ids = df["customer_id"].astype(str).tolist()
    X = df[feature_order].astype(float)

    return X, ids
