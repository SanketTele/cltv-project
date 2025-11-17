"""
features.py
Create customer-level features for CLTV modeling.

Save path:
cltv-project/src/features.py

Run:
python src/features.py

Outputs:
- data/processed/customer_features.csv
- reports/customer_features_sample.csv
"""

import os
import pandas as pd
import numpy as np
from datetime import timedelta

# ---------- CONFIG ----------
INTERIM_CSV = r"C:\Users\Sanket\Desktop\Documents\Tasks\cltv-project\data\interim\transactions_clean.csv"
PROCESSED_DIR = r"C:\Users\Sanket\Desktop\Documents\Tasks\cltv-project\data\processed"
PROCESSED_CSV = os.path.join(PROCESSED_DIR, "customer_features.csv")
REPORTS_DIR = r"C:\Users\Sanket\Desktop\Documents\Tasks\cltv-project\reports"
SAMPLE_OUT = os.path.join(REPORTS_DIR, "customer_features_sample.csv")

# If True, create a supervised target `LTV_target` by splitting on a reference date.
# The logic: reference_date = max(invoice_date) - horizon_days
# Features are computed on transactions <= reference_date.
# LTV_target is sum(order_value) for transactions > reference_date.
CREATE_TARGET = True
HORIZON_DAYS = 365  # prediction horizon in days (adjustable)
MIN_FREQ = 1  # filter customers with at least this many transactions in observation window (optional)
# ----------------------------

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)


def load_data(path):
    print(f"Loading interim transactions: {path}")
    df = pd.read_csv(path, parse_dates=['invoice_date'], low_memory=False)
    return df


def build_features(trans):
    """
    trans: dataframe with columns at least ['customer_id','invoice_date','order_value','quantity','category' optional]
    returns: customer-level dataframe
    """
    # ensure types
    trans['invoice_date'] = pd.to_datetime(trans['invoice_date'])
    trans['order_value'] = pd.to_numeric(trans['order_value'], errors='coerce').fillna(0)
    if 'quantity' in trans.columns:
        trans['quantity'] = pd.to_numeric(trans['quantity'], errors='coerce').fillna(0)

    # Aggregations
    agg = trans.groupby('customer_id').agg(
        frequency=('invoice_no' if 'invoice_no' in trans.columns else 'invoice_date', 'nunique'),
        total_spend=('order_value', 'sum'),
        avg_order_value=('order_value', 'mean'),
        first_purchase=('invoice_date', 'min'),
        last_purchase=('invoice_date', 'max'),
        unique_days=('invoice_date', lambda x: x.dt.date.nunique()),
    ).reset_index()

    # Derived features
    # snapshot_date will be set by caller (we use last_purchase reference externally if needed)
    # For now compute T and recency relative to the full dataset max date (caller may adjust)
    dataset_max = trans['invoice_date'].max()
    agg['recency_days'] = (dataset_max - agg['last_purchase']).dt.days
    agg['T_days'] = (dataset_max - agg['first_purchase']).dt.days.replace(0, 1)  # avoid div by zero
    agg['aov'] = agg['avg_order_value']
    agg['avg_interpurchase_days'] = agg['T_days'] / agg['frequency']
    # Active months
    trans['month'] = trans['invoice_date'].dt.to_period('M')
    months = trans.groupby('customer_id')['month'].nunique().reset_index().rename(columns={'month': 'active_months'})
    agg = agg.merge(months, on='customer_id', how='left')
    # Purchase days std: variation in days between purchases
    def purchase_days_std(sub):
        dates = sub.sort_values('invoice_date')['invoice_date'].drop_duplicates()
        if len(dates) <= 1:
            return 0.0
        diffs = dates.diff().dt.days.dropna()
        return float(diffs.std()) if not diffs.empty else 0.0
    purchase_std = trans.groupby('customer_id').apply(purchase_days_std).reset_index().rename(columns={0: 'purchase_days_std'})
    agg = agg.merge(purchase_std, on='customer_id', how='left')
    # Category diversity (if category exists)
    if 'category' in trans.columns:
        cat_div = trans.groupby('customer_id')['category'].nunique().reset_index().rename(columns={'category': 'category_diversity'})
        agg = agg.merge(cat_div, on='customer_id', how='left')
    else:
        agg['category_diversity'] = 0

    # Fill NaNs
    agg['purchase_days_std'] = agg['purchase_days_std'].fillna(0)
    agg['active_months'] = agg['active_months'].fillna(0).astype(int)
    return agg


def create_target_from_split(full_df, horizon_days=365):
    """
    Create features on transactions <= reference_date and create LTV_target as sum of order_value > reference_date.
    Returns feature_df (features computed on observation window) with appended LTV_target column.
    """
    max_date = full_df['invoice_date'].max()
    reference_date = max_date - pd.Timedelta(days=horizon_days)
    print(f"Max invoice date in data: {max_date.date()}")
    print(f"Reference date (end of observation window): {reference_date.date()}")
    obs_df = full_df[full_df['invoice_date'] <= reference_date].copy()
    target_df = full_df[(full_df['invoice_date'] > reference_date) & (full_df['invoice_date'] <= max_date)].copy()

    # Guard: if obs_df empty, return None
    if obs_df.empty:
        raise ValueError("Observation window is empty. Reduce HORIZON_DAYS or provide longer history.")

    features = build_features(obs_df)
    # compute LTV_target per customer from target_df
    ltv = target_df.groupby('customer_id').agg(LTV_target=('order_value', 'sum')).reset_index()
    features = features.merge(ltv, on='customer_id', how='left')
    features['LTV_target'] = features['LTV_target'].fillna(0.0)
    # Optionally filter customers with very low frequency in obs window
    features = features[features['frequency'] >= MIN_FREQ].copy()
    return features, reference_date, max_date


def main():
    df = load_data(INTERIM_CSV)

    # Basic sanity check
    if df.empty:
        raise ValueError(f"No rows found in {INTERIM_CSV}")

    # Standard column name adjustments (make robust)
    columns = [c.lower().strip() for c in df.columns]
    rename_map = {}
    for orig, c in zip(df.columns, columns):
        if 'customer' in c and 'id' in c:
            rename_map[orig] = 'customer_id'
        if 'invoice' in c and 'date' in c:
            rename_map[orig] = 'invoice_date'
        if 'invoice' in c and ('no' in c or 'num' in c):
            rename_map[orig] = 'invoice_no'
        if 'unit_price' in c or 'unit price' in c:
            rename_map[orig] = 'unit_price'
        if 'quantity' == c:
            rename_map[orig] = 'quantity'
        if 'amount' in c or 'order_value' in c or 'total' in c:
            rename_map[orig] = 'order_value'
        if 'description' in c:
            rename_map[orig] = 'description'
        if 'stockcode' in c or 'product' in c:
            rename_map[orig] = 'product_id'
        if 'category' in c:
            rename_map[orig] = 'category'
    if rename_map:
        df = df.rename(columns=rename_map)

    # Ensure required columns exist
    if 'order_value' not in df.columns:
        # attempt to compute
        if 'quantity' in df.columns and 'unit_price' in df.columns:
            df['order_value'] = df['quantity'] * df['unit_price']
        else:
            raise ValueError("order_value not found and cannot be computed. Ensure quantity & unit_price exist.")

    # Cast types
    df['invoice_date'] = pd.to_datetime(df['invoice_date'], errors='coerce')
    df = df.dropna(subset=['invoice_date', 'customer_id'])
    # Make customer_id string
    df['customer_id'] = df['customer_id'].astype(str).str.strip()

    if CREATE_TARGET:
        features, reference_date, max_date = create_target_from_split(df, horizon_days=HORIZON_DAYS)
        # Save metadata about dates
        meta = {
            'reference_date': reference_date.isoformat(),
            'max_date': max_date.isoformat(),
            'horizon_days': HORIZON_DAYS
        }
    else:
        # Create features on full dataset (no supervised target)
        features = build_features(df)
        meta = {'max_date': df['invoice_date'].max().isoformat(), 'horizon_days': None}

    # Re-order columns for readability
    cols_order = ['customer_id', 'frequency', 'total_spend', 'aov', 'recency_days', 'T_days',
                  'avg_interpurchase_days', 'active_months', 'purchase_days_std', 'category_diversity']
    existing = [c for c in cols_order if c in features.columns]
    rest = [c for c in features.columns if c not in existing]
    final_cols = existing + rest
    features = features[final_cols]

    # Save
    features.to_csv(PROCESSED_CSV, index=False)
    print(f"Saved customer features: {PROCESSED_CSV} | Rows: {len(features):,}")

    # Save sample for quick inspection
    features.head(200).to_csv(SAMPLE_OUT, index=False)
    print(f"Saved sample: {SAMPLE_OUT}")

    # Save meta
    try:
        import json
        meta_path = os.path.join(os.path.dirname(PROCESSED_CSV), "features_meta.json")
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)
        print(f"Saved meta: {meta_path}")
    except Exception:
        pass


if __name__ == "__main__":
    main()
