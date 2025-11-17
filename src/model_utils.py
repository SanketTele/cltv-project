# model_utils.py
import os
import joblib
import pandas as pd


MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "models",
    "xgb_cltv_model.joblib"
)


def load_model_and_feature_order():
    """
    Load XGBoost model and determine the correct feature order.
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

    model = joblib.load(MODEL_PATH)

    # Try to read feature names from booster
    try:
        booster = model.get_booster()
        feature_order = booster.feature_names
    except:
        feature_order = getattr(model, "feature_names_in_", None)

    # Last fallback (should rarely be used)
    if not feature_order:
        feature_order = [
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


def prepare_input_df(customers_list, feature_order):
    """
    Convert user JSON input → DataFrame with correct model feature order.
    """
    df = pd.DataFrame(customers_list)

    if "customer_id" not in df:
        raise ValueError("customer_id missing.")

    ids = df["customer_id"].astype(str).tolist()

    # Auto-generate missing columns
    if "avg_order_value" not in df.columns and "aov" in df.columns:
        df["avg_order_value"] = df["aov"]

    if "unique_days" not in df.columns:
        df["unique_days"] = df["frequency"] if "frequency" in df.columns else 0

    # Ensure all columns exist
    for col in feature_order:
        if col not in df.columns:
            df[col] = 0.0

    X = df[feature_order].astype(float).fillna(0)

    return X, ids
