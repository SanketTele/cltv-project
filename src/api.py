# src/api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import traceback
import os

import numpy as np
import xgboost as xgb
import joblib

# Helper utilities - ensure these exist in src/model_utils.py
# load_model_and_feature_order() -> (model_object, feature_order_list)
# prepare_input_df(customers_list, feature_order) -> (X_dataframe_or_numpy, ids_list)
from .model_utils import load_model_and_feature_order, prepare_input_df

app = FastAPI(title="CLTV Prediction API (XGBoost contributions)", version="1.0")

# ---------- Root ----------
@app.get("/")
def root():
    return {
        "service": "cltv-api",
        "status": "running",
        "description": "Customer Lifetime Value prediction API (uses XGBoost contributions for explanations)",
        "endpoints": {"health": "/health", "docs": "/docs", "predict (POST)": "/predict"}
    }

# ---------- Globals ----------
MODEL = None
MODEL_FEATURE_ORDER: Optional[List[str]] = None
# segmentation thresholds (fallbacks; overwritten at startup from env)
LTV_LOW_THRESHOLD = 50.0
LTV_MED_THRESHOLD = 200.0

# ---------- Pydantic models ----------
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
    return_explanation: Optional[bool] = False  # ask for per-sample contributions

class FeatureImpact(BaseModel):
    feature: str
    impact: float

class PredictResponseItem(BaseModel):
    customer_id: str
    predicted_LTV: float
    segment: Optional[str] = None
    explanation: Optional[List[FeatureImpact]] = None

# ---------- Utility functions ----------
def ltv_to_segment(ltv: float, low_threshold: float, med_threshold: float) -> str:
    if ltv < low_threshold:
        return "Low"
    if ltv < med_threshold:
        return "Medium"
    return "High"

def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default

# ---------- Startup: load model and thresholds ----------
@app.on_event("startup")
def startup_event():
    global MODEL, MODEL_FEATURE_ORDER, LTV_LOW_THRESHOLD, LTV_MED_THRESHOLD
    try:
        MODEL, MODEL_FEATURE_ORDER = load_model_and_feature_order()
        print("Model loaded. Feature order:", MODEL_FEATURE_ORDER)
    except Exception as e:
        MODEL = None
        MODEL_FEATURE_ORDER = None
        print("ERROR loading model:", e)

    # Read segmentation thresholds from environment (if present)
    try:
        LTV_LOW_THRESHOLD = safe_float(os.environ.get("LTV_LOW_THRESHOLD", LTV_LOW_THRESHOLD))
        LTV_MED_THRESHOLD = safe_float(os.environ.get("LTV_MED_THRESHOLD", LTV_MED_THRESHOLD))
    except Exception as e:
        print("Error reading thresholds from environment:", e)
    print(f"Segmentation thresholds: low={LTV_LOW_THRESHOLD}, med={LTV_MED_THRESHOLD}")

# ---------- Health ----------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": MODEL is not None,
        "feature_count": len(MODEL_FEATURE_ORDER) if MODEL_FEATURE_ORDER else 0,
        "thresholds": {"low": LTV_LOW_THRESHOLD, "med": LTV_MED_THRESHOLD}
    }

# ---------- Prediction endpoint (with XGBoost contributions) ----------
@app.post("/predict", response_model=List[PredictResponseItem])
def predict(req: PredictRequest):
    try:
        if MODEL is None or MODEL_FEATURE_ORDER is None:
            raise HTTPException(status_code=503, detail="Model not loaded on server.")

        # Prepare inputs
        customers_list = [c.dict() for c in req.customers]
        X, ids = prepare_input_df(customers_list, MODEL_FEATURE_ORDER)

        # Convert X to numpy array if it's a pandas DataFrame
        try:
            import pandas as pd
            if isinstance(X, pd.DataFrame):
                X_np = X.values
            else:
                X_np = np.array(X)
        except Exception:
            X_np = np.array(X)

        # Predict LTV
        preds = MODEL.predict(X_np)
        preds = [float(max(0.0, p)) for p in preds]  # clamp negative predictions to 0.0

        # Thresholds (allow env override per request)
        low_th = safe_float(os.environ.get("LTV_LOW_THRESHOLD", LTV_LOW_THRESHOLD))
        med_th = safe_float(os.environ.get("LTV_MED_THRESHOLD", LTV_MED_THRESHOLD))

        need_expl = bool(getattr(req, "return_explanation", False))

        # Compute contributions only if requested
        contribs = None
        if need_expl:
            try:
                # Try sklearn wrapper predict with pred_contribs flag
                try:
                    contribs_np = MODEL.predict(X_np, pred_contribs=True)
                except TypeError:
                    # Fallback: use booster + DMatrix
                    dmat = xgb.DMatrix(X_np, feature_names=MODEL_FEATURE_ORDER)
                    contribs_np = MODEL.get_booster().predict(dmat, pred_contribs=True)
                contribs = np.array(contribs_np)
                # If XGBoost returns an extra bias/expected_value column, drop it
                if contribs.ndim == 2 and contribs.shape[1] == len(MODEL_FEATURE_ORDER) + 1:
                    contribs = contribs[:, :len(MODEL_FEATURE_ORDER)]
                # If shape mismatch, set to None
                if contribs.ndim != 2 or contribs.shape[1] != len(MODEL_FEATURE_ORDER):
                    print("Unexpected contributions shape:", getattr(contribs, "shape", None))
                    contribs = None
            except Exception as e:
                print("XGBoost contributions failed:", e)
                contribs = None

        # Build response
        results: List[Dict[str, Any]] = []
        for i, (cid, p) in enumerate(zip(ids, preds)):
            seg = ltv_to_segment(p, low_th, med_th)
            item: Dict[str, Any] = {"customer_id": cid, "predicted_LTV": p, "segment": seg}

            if need_expl and contribs is not None:
                row = contribs[i]
                abs_impacts = np.abs(row)
                k = 3
                top_idx = abs_impacts.argsort()[::-1][:k]
                top_list = []
                for idx in top_idx:
                    feat = MODEL_FEATURE_ORDER[idx]
                    impact = float(row[idx])
                    top_list.append({"feature": feat, "impact": impact})
                item["explanation"] = top_list

            results.append(item)

        return results

    except HTTPException:
        raise
    except Exception:
        traceback_str = traceback.format_exc()
        raise HTTPException(status_code=500, detail=traceback_str)
