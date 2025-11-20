# src/api.py
"""
Debuggable CLTV Prediction API with SHAP explainability.
Overwrite this exact file at:
C:\Users\Sanket\Desktop\Documents\Tasks\cltv-project\src\api.py
"""

import os
import traceback
from typing import List, Optional, Dict, Any

import numpy as np
import pandas as pd
import joblib
import shap

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Local helpers (ensure src/model_utils.py exists and matches feature order)
from .model_utils import load_model_and_feature_order, prepare_input_df

app = FastAPI(title="CLTV API (debug SHAP)",
              description="CLTV API with verbose logs for SHAP debugging",
              version="1.0")

# Globals to be populated at startup
MODEL = None
MODEL_FEATURE_ORDER: Optional[List[str]] = None
EXPLAINER = None
LTV_LOW_THRESHOLD = 50.0
LTV_MED_THRESHOLD = 200.0

# ----------------------------
# Schemas
# ----------------------------
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

# ----------------------------
# Utilities
# ----------------------------
def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default

def ltv_to_segment(ltv: float, low: float, med: float) -> str:
    if ltv < low:
        return "Low"
    if ltv < med:
        return "Medium"
    return "High"

def _debug_print(title: str, obj: Any = None):
    """Small helper for consistent debug prints in logs."""
    print("==== DEBUG:", title, "====")
    if obj is not None:
        try:
            print(obj)
        except Exception:
            print("Could not print object (type):", type(obj))
    print("==== /DEBUG ====")

# ----------------------------
# Startup: load model, thresholds, build SHAP explainer
# ----------------------------
@app.on_event("startup")
def startup_event():
    global MODEL, MODEL_FEATURE_ORDER, EXPLAINER, LTV_LOW_THRESHOLD, LTV_MED_THRESHOLD
    try:
        MODEL, MODEL_FEATURE_ORDER = load_model_and_feature_order()
        print("Model loaded successfully. Feature order:")
        _debug_print("MODEL_FEATURE_ORDER", MODEL_FEATURE_ORDER)
    except Exception as e:
        MODEL = None
        MODEL_FEATURE_ORDER = None
        print("ERROR: Failed to load model in startup_event.")
        print(traceback.format_exc())

    # read thresholds from env if present
    try:
        LTV_LOW_THRESHOLD = safe_float(os.environ.get("LTV_LOW_THRESHOLD", LTV_LOW_THRESHOLD))
        LTV_MED_THRESHOLD = safe_float(os.environ.get("LTV_MED_THRESHOLD", LTV_MED_THRESHOLD))
        print(f"Segmentation thresholds loaded: low={LTV_LOW_THRESHOLD}, med={LTV_MED_THRESHOLD}")
    except Exception:
        print("WARNING: Error reading threshold env vars:")
        print(traceback.format_exc())

    # Build SHAP TreeExplainer
    EXPLAINER = None
    try:
        if MODEL is not None:
            EXPLAINER = shap.TreeExplainer(MODEL)
            print("SHAP TreeExplainer created successfully.")
        else:
            print("Model not loaded; skipping SHAP explainer creation.")
    except Exception as e:
        EXPLAINER = None
        print("SHAP explainer creation FAILED during startup.")
        print(traceback.format_exc())

# ----------------------------
# Health endpoint
# ----------------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": MODEL is not None,
        "explainer_loaded": EXPLAINER is not None,
        "feature_count": len(MODEL_FEATURE_ORDER) if MODEL_FEATURE_ORDER else 0,
        "thresholds": {"low": LTV_LOW_THRESHOLD, "med": LTV_MED_THRESHOLD}
    }

# ----------------------------
# Predict endpoint (verbose)
# ----------------------------
@app.post("/predict", response_model=List[PredictResponseItem])
def predict(req: PredictRequest):
    try:
        if MODEL is None or MODEL_FEATURE_ORDER is None:
            raise HTTPException(status_code=503, detail="Model not loaded on server.")

        customers_list = [c.dict() for c in req.customers]
        X, ids = prepare_input_df(customers_list, MODEL_FEATURE_ORDER)

        # Convert to DataFrame if not already
        if not isinstance(X, pd.DataFrame):
            try:
                X = pd.DataFrame(X, columns=MODEL_FEATURE_ORDER)
            except Exception:
                X = pd.DataFrame(X)

        _debug_print("Input DataFrame shape and head", {"shape": X.shape, "head": X.head(3).to_dict()})

        preds_raw = MODEL.predict(X)
        preds = [float(max(0.0, p)) for p in preds_raw]

        low_th = safe_float(os.environ.get("LTV_LOW_THRESHOLD", LTV_LOW_THRESHOLD))
        med_th = safe_float(os.environ.get("LTV_MED_THRESHOLD", LTV_MED_THRESHOLD))

        need_expl = bool(req.return_explanation)
        _debug_print("Request flags", {"return_explanation": need_expl})

        shap_values = None
        if need_expl:
            if EXPLAINER is None:
                print("Client requested explanation but EXPLAINER is None (not built).")
            else:
                try:
                    # Compute SHAP explanations
                    expl_res = EXPLAINER(X)
                    if hasattr(expl_res, "values"):
                        shap_values = np.array(expl_res.values)
                    else:
                        shap_values = np.array(expl_res)
                    _debug_print("shap_values shape", getattr(shap_values, "shape", None))
                    # Validate shape
                    if shap_values.ndim != 2 or shap_values.shape[1] != len(MODEL_FEATURE_ORDER):
                        print("Unexpected shap_values shape. Disabling explanations for this request.")
                        _debug_print("shap_values full", shap_values if shap_values is not None else "None")
                        shap_values = None
                except Exception:
                    print("SHAP computation FAILED at runtime for this request:")
                    print(traceback.format_exc())
                    shap_values = None

        # Build response
        results: List[Dict[str, Any]] = []
        for i, cid in enumerate(ids):
            pred = preds[i]
            seg = ltv_to_segment(pred, low_th, med_th)
            item: Dict[str, Any] = {"customer_id": cid, "predicted_LTV": pred, "segment": seg}

            if shap_values is not None:
                row = shap_values[i]
                abs_vals = np.abs(row)
                k = 3
                top_idx = abs_vals.argsort()[::-1][:k]
                explanation_list = []
                for idx in top_idx:
                    explanation_list.append({"feature": MODEL_FEATURE_ORDER[idx], "impact": float(row[idx])})
                item["explanation"] = explanation_list

            results.append(item)

        _debug_print("Response preview", results[:2])
        return results

    except HTTPException:
        raise
    except Exception:
        traceback_str = traceback.format_exc()
        print("Unhandled exception in /predict:")
        print(traceback_str)
        raise HTTPException(status_code=500, detail=traceback_str)
