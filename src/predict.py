"""
predict.py
Generate predictions for all customers, segment them, and save reports.

Run:
python src\predict.py

Outputs:
- reports/cltv_predictions_full.csv
- reports/cltv_top_customers.csv
- reports/cltv_segment_summary.csv
"""

import os
import pandas as pd
import numpy as np
import joblib

# Config
FEATURES_CSV = r"C:\Users\Sanket\Desktop\Documents\Tasks\cltv-project\data\processed\customer_features.csv"
MODEL_PATH = r"C:\Users\Sanket\Desktop\Documents\Tasks\cltv-project\models\xgb_cltv_model.joblib"
REPORTS_DIR = r"C:\Users\Sanket\Desktop\Documents\Tasks\cltv-project\reports"
TOP_N = 200  # top customers to save

os.makedirs(REPORTS_DIR, exist_ok=True)
OUT_FULL = os.path.join(REPORTS_DIR, "cltv_predictions_full.csv")
OUT_TOP = os.path.join(REPORTS_DIR, "cltv_top_customers.csv")
OUT_SUMMARY = os.path.join(REPORTS_DIR, "cltv_segment_summary.csv")

def load_artifacts():
    print("Loading features and model...")
    features = pd.read_csv(FEATURES_CSV)
    model = joblib.load(MODEL_PATH)
    return features, model

def prepare_X(features):
    ignore_cols = ['customer_id', 'first_purchase', 'last_purchase', 'LTV_target']
    X = features.copy()
    for c in ignore_cols:
        if c in X.columns:
            X = X.drop(columns=[c])
    # Keep customer_id separately
    cust_ids = features['customer_id'].astype(str).tolist()
    X = X.fillna(0)
    return X, cust_ids

def predict_all(model, X):
    print("Predicting LTV for all customers...")
    preds = model.predict(X)
    preds = np.maximum(preds, 0.0)  # force non-negative
    return preds

def make_segments(df, score_col='predicted_LTV'):
    # Default segmentation: Low (bottom 60%), Medium (60-90), High (top 10%)
    df['segment'] = pd.qcut(df[score_col], q=[0, 0.6, 0.9, 1.0], labels=['Low','Medium','High'])
    return df

def try_shap_explanations(model, X, features_df, top_k=3):
    """
    Compute SHAP values if shap is installed and model supports it.
    Returns a dict customer_id -> list of (feature, shap_value) for top_k features.
    If shap not available or fails, returns empty dict.
    """
    try:
        import shap
        print("Computing SHAP values (this may take a while for large datasets)...")
        explainer = shap.Explainer(model)
        shap_values = explainer(X)  # may be a shap.Explanation object
        # shap_values.values: (n_samples, n_features)
        values = shap_values.values
        feature_names = X.columns.tolist()
        drivers = {}
        for i, cust in enumerate(features_df['customer_id'].astype(str).tolist()):
            row = values[i]
            top_idx = np.argsort(-np.abs(row))[:top_k]
            drivers[cust] = [(feature_names[j], float(row[j])) for j in top_idx]
        return drivers
    except Exception as e:
        print("SHAP not computed (missing or failed). Reason:", str(e))
        return {}

def drivers_to_text(drivers_dict, cust_id):
    if not drivers_dict:
        return ""
    items = drivers_dict.get(str(cust_id), [])
    return "; ".join([f"{feat}:{val:.2f}" for feat,val in items])

def main():
    features, model = load_artifacts()
    X, cust_ids = prepare_X(features)
    preds = predict_all(model, X)

    # Create results dataframe
    result = features[['customer_id']].copy()
    result['predicted_LTV'] = preds

    # Segmentation
    result = make_segments(result, score_col='predicted_LTV')

    # Try SHAP explanations (optional)
    shap_drivers = try_shap_explanations(model, X, features, top_k=3)

    # Add driver text column (if shap succeeded)
    if shap_drivers:
        result['top_3_drivers'] = result['customer_id'].astype(str).apply(lambda cid: drivers_to_text(shap_drivers, cid))
    else:
        result['top_3_drivers'] = ""

    # Save full predictions
    result.to_csv(OUT_FULL, index=False)
    print(f"Saved full predictions: {OUT_FULL}")

    # Save top-N customers
    top_df = result.sort_values('predicted_LTV', ascending=False).head(TOP_N)
    top_df.to_csv(OUT_TOP, index=False)
    print(f"Saved top {TOP_N} customers: {OUT_TOP}")

    # Segment summary KPIs
    merged = features.merge(result[['customer_id','predicted_LTV','segment']], on='customer_id', how='left')
    summary = merged.groupby('segment').agg(
        customers=('customer_id','nunique'),
        avg_predicted_LTV=('predicted_LTV','mean'),
        avg_total_spend=('total_spend','mean') if 'total_spend' in merged.columns else ('predicted_LTV','mean'),
        avg_freq=('frequency','mean') if 'frequency' in merged.columns else (pd.NamedAgg(column='customer_id', aggfunc='count'))
    ).reset_index()
    summary.to_csv(OUT_SUMMARY, index=False)
    print(f"Saved segment summary: {OUT_SUMMARY}")

    print("Done.")

if __name__ == "__main__":
    main()
