# src/api.py
"""
CLTV Prediction API with SHAP explanations (TreeExplainer).
Save as: C:\Users\Sanket\Desktop\Documents\Tasks\cltv-project\src\api.py
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import traceback
import os
import numpy as np
import pandas as pd
import joblib

# SHAP (installed inside Docker)
import shap

# Local helpers (must exist at src/model_utils.py)
from .model_utils import load_model_and_feature_order, prepare_input_df

app = FastAPI(
    title="CLTV Prediction API with SHAP",
    description="Predict Customer Lifetime Value and return SHAP explanations (TreeExplainer)",
    version="1.0"
)

# ---------- Root ----------
@app.get("/")
def root():
    return {
        "service": "cltv-api",
        "status": "running",
        "description": "Customer Lifetime Value prediction API with SHAP",
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "predict (POST)": "/predict"
        }
    }

# ---------- Globals (populated at startup) ----------
MODEL = None
MODEL_FEATURE_ORDER: Optional[List[str]] = None
EXPLAINER = None

# default thresholds (can be overridden by env vars)
LTV_LOW_THRESHOLD = 50.0
LTV_MED_THRESHOLD = 200.0

# ---------- Pydantic schemas ----------
class CustomerFeatures(BaseModel):
    customer_id: str = Field(..., example="C101")
    frequency: float = 0.0
    total_spend: float = 0.0
    aov: float = 0.0
    recency_days: float = 0.0
    T_days: float = 0.0
    avg_interpurchase_days: float = 0.0
    active_months: float = 0.0
    purchase_days_std: float = 0.0
    category_diversity: float = 0.0
    avg_order_value: float = 0.0
    unique_days: float = 0.0

class PredictRequest(BaseModel):
    customers: List[CustomerFeatures]
    return_explanation: Optional[bool] = False

class FeatureImpact(BaseModel):
    feature: str
    impact: float

class PredictResponseItem(BaseModel):
    customer_id: str
    predicted_LTV: float
    segment: Optional[str]
    explanation: Optional[List[FeatureImpact]]

# ---------- Utilities ----------
def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default

def ltv_to_segment(ltv: float, low_threshold: float, med_threshold: float) -> str:
    if ltv < low_threshold:
        return "Low"
    if ltv < med_threshold:
        return "Medium"
    return "High"

# ---------- Startup: load model & build SHAP explainer ----------
@app.on_event("startup")
def startup_event():
    global MODEL, MODEL_FEATURE_ORDER, EXPLAINER, LTV_LOW_THRESHOLD, LTV_MED_THRESHOLD

    try:
        MODEL, MODEL_FEATURE_ORDER = load_model_and_feature_order()
        print("Model loaded. Features:", MODEL_FEATURE_ORDER)
    except Exception as e:
        MODEL = None
        MODEL_FEATURE_ORDER = None
        print("ERROR loading model:", e)
        print(traceback.format_exc())

    # read thresholds from environment if provided
    try:
        LTV_LOW_THRESHOLD = safe_float(os.environ.get("LTV_LOW_THRESHOLD", LTV_LOW_THRESHOLD))
        LTV_MED_THRESHOLD = safe_float(os.environ.get("LTV_MED_THRESHOLD", LTV_MED_THRESHOLD))
    except Exception as e:
        print("Error parsing thresholds:", e)

    print(f"Segmentation thresholds: low={LTV_LOW_THRESHOLD}, med={LTV_MED_THRESHOLD}")

    # Build SHAP TreeExplainer for tree-based model (XGBoost LightGBM)
    EXPLAINER = None
    try:
        if MODEL is not None:
            # TreeExplainer works for xgboost.Booster or sklearn wrapper
            EXPLAINER = shap.TreeExplainer(MODEL)
            print("SHAP TreeExplainer created successfully.")
        else:
            print("Model not loaded; cannot build SHAP explainer.")
    except Exception as e:
        EXPLAINER = None
        print("SHAP explainer creation failed:", e)
        print(traceback.format_exc())

# ---------- Health endpoint ----------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": MODEL is not None,
        "explainer_loaded": EXPLAINER is not None,
        "feature_count": len(MODEL_FEATURE_ORDER) if MODEL_FEATURE_ORDER else 0,
        "thresholds": {"low": LTV_LOW_THRESHOLD, "med": LTV_MED_THRESHOLD}
    }

# ---------- Prediction endpoint ----------
@app.post("/predict", response_model=List[PredictResponseItem])
def predict(req: PredictRequest):
    try:
        if MODEL is None or MODEL_FEATURE_ORDER is None:
            raise HTTPException(status_code=503, detail="Model not loaded on server.")

        # prepare inputs; prepare_input_df returns (X, ids)
        customers_list = [c.dict() for c in req.customers]
        X, ids = prepare_input_df(customers_list, MODEL_FEATURE_ORDER)

        # ensure X is pandas DataFrame (SHAP/Model support)
        if not isinstance(X, pd.DataFrame):
            try:
                X = pd.DataFrame(X, columns=MODEL_FEATURE_ORDER)
            except Exception:
                X = pd.DataFrame(X)

        # Predictions
        preds_raw = MODEL.predict(X)
        preds = [float(max(0.0, p)) for p in preds_raw]

        # thresholds (allow environment override)
        low_th = safe_float(os.environ.get("LTV_LOW_THRESHOLD", LTV_LOW_THRESHOLD))
        med_th = safe_float(os.environ.get("LTV_MED_THRESHOLD", LTV_MED_THRESHOLD))

        need_expl = bool(req.return_explanation) and EXPLAINER is not None

        # compute shap values if requested
        shap_values = None
        if need_expl:
            try:
                # EXPLAINER(X) returns Explanation; shap_values often in .values
                expl_res = EXPLAINER(X)
                if hasattr(expl_res, "values"):
                    shap_values = np.array(expl_res.values)
                else:
                    shap_values = np.array(expl_res)  # fallback
                # shap_values shape -> (n_samples, n_features)
                if shap_values.ndim != 2 or shap_values.shape[1] != len(MODEL_FEATURE_ORDER):
                    # Unexpected shape; log and disable explanations
                    print("Unexpected shap_values shape:", getattr(shap_values, "shape", None))
                    shap_values = None
            except Exception as e:
                print("SHAP computation failed:", e)
                print(traceback.format_exc())
                shap_values = None
                need_expl = False

        # Build response list
        results: List[Dict[str, Any]] = []
        for i, cid in enumerate(ids):
            pred = preds[i]
            seg = ltv_to_segment(pred, low_th, med_th)
            item: Dict[str, Any] = {
                "customer_id": cid,
                "predicted_LTV": pred,
                "segment": seg
            }

            if need_expl and shap_values is not None:
                row = shap_values[i]
                abs_vals = np.abs(row)
                k = 3
                top_idx = abs_vals.argsort()[::-1][:k]
                explanation_list = []
                for idx in top_idx:
                    feat = MODEL_FEATURE_ORDER[idx]
                    impact = float(row[idx])
                    explanation_list.append({"feature": feat, "impact": impact})
                item["explanation"] = explanation_list

            results.append(item)

        return results

    except HTTPException:
        raise
    except Exception:
        traceback_str = traceback.format_exc()
        raise HTTPException(status_code=500, detail=traceback_str)
