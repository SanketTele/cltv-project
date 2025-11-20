# src/api.py
"""
CLTV Prediction API with robust SHAP explainer builder.
This file does NOT contain any Windows backslash paths to avoid unicodeescape issues.
"""

import os
import traceback
from typing import List, Optional, Dict, Any

import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# shap may be installed in Docker environment; handle if missing locally
try:
    import shap  # type: ignore
except Exception:
    shap = None

# local helpers
from .model_utils import load_model_and_feature_order, prepare_input_df

app = FastAPI(title="CLTV Prediction API (SHAP robust)", version="1.0")

# Globals
MODEL = None
MODEL_FEATURE_ORDER: Optional[List[str]] = None
EXPLAINER = None
LTV_LOW_THRESHOLD = 50.0
LTV_MED_THRESHOLD = 200.0

# Pydantic schemas
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

# Utilities
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
    print("==== DEBUG:", title, "====")
    if obj is not None:
        try:
            print(obj)
        except Exception:
            print("Could not print debug object of type:", type(obj))
    print("==== /DEBUG ====")

# Robust SHAP explainer builder
def build_shap_explainer(model, sample_X=None):
    """
    Try multiple ways to construct a SHAP explainer:
    1) If model has get_booster() or booster_ -> use Booster with TreeExplainer.
    2) Try TreeExplainer(model).
    3) Fallback to shap.Explainer(model, data=sample_X).
    """
    if shap is None:
        print("SHAP is not installed in runtime.")
        return None

    booster = None
    try:
        if hasattr(model, "get_booster"):
            try:
                booster = model.get_booster()
                _debug_print("Using model.get_booster()", str(type(booster)))
            except Exception:
                booster = None
        elif hasattr(model, "booster_"):
            try:
                booster = model.booster_
                _debug_print("Using model.booster_", str(type(booster)))
            except Exception:
                booster = None
    except Exception:
        booster = None

    if booster is not None:
        try:
            expl = shap.TreeExplainer(booster)
            print("SHAP TreeExplainer built from Booster successfully.")
            return expl
        except Exception as e:
            print("Failed to build TreeExplainer from Booster:", e)
            print("Traceback:", traceback.format_exc())

    try:
        expl = shap.TreeExplainer(model)
        print("SHAP TreeExplainer built from model successfully.")
        return expl
    except Exception as e:
        print("Failed to build TreeExplainer from model:", e)
        print("Traceback:", traceback.format_exc())

    try:
        if sample_X is not None:
            expl = shap.Explainer(model, sample_X)
            print("SHAP generic Explainer built with sample data (fallback).")
            return expl
        else:
            expl = shap.Explainer(model)
            print("SHAP generic Explainer built without sample data (fallback).")
            return expl
    except Exception as e:
        print("Failed to build generic SHAP Explainer:", e)
        print("Traceback:", traceback.format_exc())

    return None

# Startup event: load model, thresholds, build explainer
@app.on_event("startup")
def startup_event():
    global MODEL, MODEL_FEATURE_ORDER, EXPLAINER, LTV_LOW_THRESHOLD, LTV_MED_THRESHOLD

    try:
        MODEL, MODEL_FEATURE_ORDER = load_model_and_feature_order()
        print("Model loaded. Feature order:")
        _debug_print("MODEL_FEATURE_ORDER", MODEL_FEATURE_ORDER)
    except Exception:
        MODEL = None
        MODEL_FEATURE_ORDER = None
        print("ERROR: Failed to load model at startup.")
        print(traceback.format_exc())

    LTV_LOW_THRESHOLD = safe_float(os.environ.get("LTV_LOW_THRESHOLD", LTV_LOW_THRESHOLD))
    LTV_MED_THRESHOLD = safe_float(os.environ.get("LTV_MED_THRESHOLD", LTV_MED_THRESHOLD))
    print(f"Segmentation thresholds: low={LTV_LOW_THRESHOLD}, med={LTV_MED_THRESHOLD}")

    EXPLAINER = None
    if MODEL is None:
        print("Model not loaded; SKIPPING SHAP explainer creation.")
        return

    sample_X = None
    try:
        sample_X = pd.DataFrame([ {f: 0.0 for f in MODEL_FEATURE_ORDER} ])
    except Exception:
        sample_X = None

    print("Attempting to build SHAP explainer (robust).")
    EXPLAINER = build_shap_explainer(MODEL, sample_X=sample_X)
    print("EXPLAINER built:", EXPLAINER is not None)

# Health
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": MODEL is not None,
        "explainer_loaded": EXPLAINER is not None,
        "feature_count": len(MODEL_FEATURE_ORDER) if MODEL_FEATURE_ORDER else 0,
        "thresholds": {"low": LTV_LOW_THRESHOLD, "med": LTV_MED_THRESHOLD}
    }

# Predict endpoint
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

        _debug_print("Input shape and head", {"shape": X.shape, "head": X.head(2).to_dict()})

        preds_raw = MODEL.predict(X)
        preds = [float(max(0.0, p)) for p in preds_raw]

        low_th = safe_float(os.environ.get("LTV_LOW_THRESHOLD", LTV_LOW_THRESHOLD))
        med_th = safe_float(os.environ.get("LTV_MED_THRESHOLD", LTV_MED_THRESHOLD))

        need_expl = bool(req.return_explanation)
        shap_values = None

        if need_expl:
            if EXPLAINER is None:
                print("Explanation requested but EXPLAINER is None.")
            else:
                try:
                    expl_res = EXPLAINER(X)
                    if hasattr(expl_res, "values"):
                        shap_values = np.array(expl_res.values)
                    else:
                        shap_values = np.array(expl_res)
                    _debug_print("shap_values shape", getattr(shap_values, "shape", None))
                    if shap_values.ndim != 2 or shap_values.shape[1] != len(MODEL_FEATURE_ORDER):
                        print("Unexpected shap_values shape; disabling explanations for this request.")
                        _debug_print("shap_values sample", shap_values[:2] if shap_values is not None else None)
                        shap_values = None
                except Exception:
                    print("SHAP computation failed at runtime:")
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

        _debug_print("Response preview", results[:2])
        return results

    except HTTPException:
        raise
    except Exception:
        tb = traceback.format_exc()
        print("Unhandled exception in /predict:", tb)
        raise HTTPException(status_code=500, detail=tb)
