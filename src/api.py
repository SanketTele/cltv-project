# src/api.py
# Clean, safe CLTV API file (no raw Windows paths).
# Overwrite this file at: C:\Users\Sanket\Desktop\Documents\Tasks\cltv-project\src\api.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import traceback
import os
import numpy as np
import pandas as pd

# optional: shap (only available if installed in runtime)
try:
    import shap  # type: ignore
except Exception:
    shap = None

from .model_utils import load_model_and_feature_order, prepare_input_df

app = FastAPI(title="CLTV Prediction API", version="1.0")

MODEL = None
MODEL_FEATURE_ORDER: Optional[List[str]] = None
EXPLAINER = None

LTV_LOW_THRESHOLD = 50.0
LTV_MED_THRESHOLD = 200.0

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

def _log_debug(title: str, obj: Any = None):
    print("---- DEBUG:", title, "----")
    if obj is not None:
        try:
            print(obj)
        except Exception:
            print("Could not print debug object of type:", type(obj))
    print("---- /DEBUG ----")

@app.on_event("startup")
def startup_event():
    global MODEL, MODEL_FEATURE_ORDER, EXPLAINER, LTV_LOW_THRESHOLD, LTV_MED_THRESHOLD
    try:
        MODEL, MODEL_FEATURE_ORDER = load_model_and_feature_order()
        print("Model loaded. Feature order:")
        _log_debug("MODEL_FEATURE_ORDER", MODEL_FEATURE_ORDER)
    except Exception:
        MODEL = None
        MODEL_FEATURE_ORDER = None
        print("ERROR: Failed to load model at startup.")
        print(traceback.format_exc())

    LTV_LOW_THRESHOLD = safe_float(os.environ.get("LTV_LOW_THRESHOLD", LTV_LOW_THRESHOLD))
    LTV_MED_THRESHOLD = safe_float(os.environ.get("LTV_MED_THRESHOLD", LTV_MED_THRESHOLD))
    print(f"Segmentation thresholds set: low={LTV_LOW_THRESHOLD}, med={LTV_MED_THRESHOLD}")

    EXPLAINER = None
    if shap is None:
        print("SHAP package not available. Explanations disabled.")
    else:
        try:
            if MODEL is not None:
                EXPLAINER = shap.TreeExplainer(MODEL)
                print("SHAP TreeExplainer built successfully.")
            else:
                print("Model not loaded; cannot build SHAP explainer.")
        except Exception:
            EXPLAINER = None
            print("Failed to build SHAP TreeExplainer:")
            print(traceback.format_exc())

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": MODEL is not None,
        "explainer_loaded": EXPLAINER is not None,
        "feature_count": len(MODEL_FEATURE_ORDER) if MODEL_FEATURE_ORDER else 0,
        "thresholds": {"low": LTV_LOW_THRESHOLD, "med": LTV_MED_THRESHOLD}
    }

@app.post("/predict", response_model=List[PredictResponseItem])
def predict(req: PredictRequest):
    try:
        if MODEL is None or MODEL_FEATURE_ORDER is None:
            raise HTTPException(status_code=503, detail="Model not loaded on server.")

        customers_list = [c.dict() for c in req.customers]
        X, ids = prepare_input_df(customers_list, MODEL_FEATURE_ORDER)

        if not isinstance(X, pd.DataFrame):
            try:
                X = pd.DataFrame(X, columns=MODEL_FEATURE_ORDER)
            except Exception:
                X = pd.DataFrame(X)

        _log_debug("Input shape", {"shape": X.shape, "columns": X.columns.tolist()})

        preds_raw = MODEL.predict(X)
        preds = [float(max(0.0, p)) for p in preds_raw]

        low_th = safe_float(os.environ.get("LTV_LOW_THRESHOLD", LTV_LOW_THRESHOLD))
        med_th = safe_float(os.environ.get("LTV_MED_THRESHOLD", LTV_MED_THRESHOLD))

        shap_values = None
        if bool(req.return_explanation):
            if EXPLAINER is None:
                print("Explanation requested but EXPLAINER is not available.")
            else:
                try:
                    expl_res = EXPLAINER(X)
                    if hasattr(expl_res, "values"):
                        shap_values = np.array(expl_res.values)
                    else:
                        shap_values = np.array(expl_res)
                    _log_debug("shap_values shape", getattr(shap_values, "shape", None))
                    if shap_values.ndim != 2 or shap_values.shape[1] != len(MODEL_FEATURE_ORDER):
                        print("Unexpected SHAP shape; disabling explanations for this request.")
                        _log_debug("shap_values full", shap_values)
                        shap_values = None
                except Exception:
                    print("SHAP computation failed for this request:")
                    print(traceback.format_exc())
                    shap_values = None

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

        _log_debug("Response preview", results[:2])
        return results

    except HTTPException:
        raise
    except Exception:
        traceback_str = traceback.format_exc()
        print("Unhandled exception in /predict:")
        print(traceback_str)
        raise HTTPException(status_code=500, detail=traceback_str)
