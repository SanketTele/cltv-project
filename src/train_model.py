"""
train_model.py
Train an XGBoost regression model for Customer Lifetime Value prediction.

Run:
python src/train_model.py

Inputs:
- data/processed/customer_features.csv (from features.py)

Outputs:
- models/xgb_cltv_model.joblib
- reports/cltv_predictions.csv
- reports/feature_importance.csv
"""

import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

# ------------------- CONFIG --------------------
FEATURES_CSV = r"C:\Users\Sanket\Desktop\Documents\Tasks\cltv-project\data\processed\customer_features.csv"
MODEL_DIR = r"C:\Users\Sanket\Desktop\Documents\Tasks\cltv-project\models"
REPORTS_DIR = r"C:\Users\Sanket\Desktop\Documents\Tasks\cltv-project\reports"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, "xgb_cltv_model.joblib")
PRED_OUT = os.path.join(REPORTS_DIR, "cltv_predictions.csv")
FEAT_IMP_OUT = os.path.join(REPORTS_DIR, "feature_importance.csv")
# ------------------------------------------------


def load_data():
    df = pd.read_csv(FEATURES_CSV)
    print(f"Loaded features: {df.shape[0]} rows, {df.shape[1]} columns")

    # Check for target
    if "LTV_target" not in df.columns:
        raise ValueError("LTV_target is missing. Ensure CREATE_TARGET=True in features.py")

    return df


def train_model(df):
    # Select features
    ignore_cols = ["customer_id", "first_purchase", "last_purchase"]
    X = df.drop(columns=[c for c in ignore_cols if c in df.columns])

    if "LTV_target" not in X.columns:
        raise ValueError("Target LTV_target not present for training.")

    y = X["LTV_target"]
    X = X.drop(columns=["LTV_target"])

    # Basic cleaning
    X = X.fillna(0)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("Training XGBoost model...")

    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42
    )

    model.fit(X_train, y_train)

    # Predictions
    preds = model.predict(X_test)

    # Metrics
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    print("\nMODEL PERFORMANCE")
    print("---------------------------")
    print(f"MAE :  {mae:.2f}")
    print(f"RMSE:  {rmse:.2f}")
    print(f"R²   :  {r2:.4f}")

    # Save model
    joblib.dump(model, MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")

    # Save predictions
    out_df = pd.DataFrame({
        "actual_LTV": y_test,
        "predicted_LTV": preds
    })
    out_df.to_csv(PRED_OUT, index=False)
    print(f"Predictions saved to {PRED_OUT}")

    # Feature importance
    importance = pd.DataFrame({
        "feature": X.columns,
        "importance": model.feature_importances_
    }).sort_values(by="importance", ascending=False)

    importance.to_csv(FEAT_IMP_OUT, index=False)
    print(f"Feature importance saved to {FEAT_IMP_OUT}")

    return model


def main():
    df = load_data()
    train_model(df)


if __name__ == "__main__":
    main()
