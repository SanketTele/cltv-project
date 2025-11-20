# src/model_utils.py
"""
Model utility functions used by api.py.
Saves paths using os.path to avoid Windows backslash unicode issues.
"""

import os
import joblib
import pandas as pd
from typing import List, Tuple, Any

def load_model_and_feature_order():
    """
    Loads the model from ../models/xgb_cltv_model.joblib and returns (model, feature_order).
    Uses os.path.join for safe path handling across OS.
    """
    base_dir = os.path.dirname(__file__)  # src/
    model_path = os.path.join(base_dir, "..", "models", "xgb_cltv_model.joblib")
    model_path = os.path.normpath(model_path)  # normalize path

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    model = joblib.load(model_path)

    # IMPORTANT: this must match the features used during training (order matters)
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
    """
    customers: list of dicts, each must include 'customer_id' plus feature keys.
    feature_order: list of feature column names in the expected order.

    Returns: (X, ids) where X is a pandas DataFrame with columns=feature_order (ordered),
             and ids is a list of customer_id in same order.
    """
    df = pd.DataFrame(customers)

    # Ensure required columns exist; if missing, fill with zeros
    required_cols = ["customer_id"] + feature_order
    for col in required_cols:
        if col not in df.columns:
            # fill missing column with zeros (float)
            df[col] = 0.0

    # Re-order columns safely
    df = df[["customer_id"] + feature_order].copy()

    ids = df["customer_id"].astype(str).tolist()
    X = df[feature_order].astype(float)

    return X, ids
