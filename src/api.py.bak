
# src/api.py
"""
CLTV Prediction API with forced contribs toggle.
Set environment variable FORCE_USE_CONTRIBS=true to always use XGBoost pred_contribs fallback.
Overwrite your existing src/api.py with this file when ready.
"""

import os
import traceback
from typing import List, Optional, Dict, Any

import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Try to import shap (may not be installed)
try:
    import shap  # type: ignore
except Exception:
    shap = None

# xgboost required for pred_contribs fallback
try:
    import xgboost as xgb
except Exception:
    xgb = None

from .model_utils import load_model_and_feature_order, prepare_input_df

app = FastAPI(title="CLTV Prediction API (force contribs toggle)", version="1.0")

# Globals
MODEL = None
MODEL_FEATURE_ORDER: Optional[List[str]] = None
EXPLAINER = None

# Read toggle from env early
USE_CONTRIBS_FORCED = os.environ.get("FORCE_USE_CONTRIBS", "false").lower() in ("1", "true", "yes")

LTV_LOW_THRESHOLD = 50.0
LTV_MED_THRESHOLD = 200.0

# Schemas
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

# SHAP explainer builder (tries robustly)
def build_shap_explainer(model, sample_X=None):
    if shap is None:
        print("SHAP not installed; cannot build explainer.")
        return None

    booster = None
    try:
        if hasattr(model, "get_booster"):
            booster = model.get_booster()
        elif hasattr(model, "booster_"):
            booster = model.booster_
    except Exception:
        booster = None

    if booster is not None:
        try:
            expl = shap.TreeExplainer(booster)
            print("SHAP TreeExplainer built from Booster.")
            return expl
        except Exception:
            print("TreeExplainer from Booster failed:", traceback.format_exc())

    try:
        expl = shap.TreeExplainer(model)
        print("SHAP TreeExplainer built from model.")
        return expl
    except Exception:
        print("TreeExplainer from model failed:", traceback.format_exc())

    try:
        if sample_X is not None:
            expl = shap.Explainer(model, sample_X)
            print("SHAP generic Explainer built with sample_X.")
            return expl
        else:
            expl = shap.Explainer(model)
            print("SHAP generic Explainer built without sample_X.")
            return expl
    except Exception:
        print("Generic SHAP Explainer failed:", traceback.format_exc())

    return None

# XGBoost pred_contribs computation helper
def compute_xgb_contribs(model, X_df):
    try:
        X_np = X_df.values if isinstance(X_df, pd.DataFrame) else np.array(X_df)
        booster = None
        try:
            if hasattr(model, "get_booster"):
                booster = model.get_booster()
            elif hasattr(model, "booster_"):
                booster = model.booster_
        except Exception:
            booster = None

        if booster is not None and xgb is not None:
            dmat = xgb.DMatrix(X_np, feature_names=list(X_df.columns) if hasattr(X_df, "columns") else None)
            contribs = booster.predict(dmat, pred_contribs=True)
            contribs = np.array(contribs)
            if contribs.ndim == 2 and contribs.shape[1] == len(X_df.columns) + 1:
                contribs = contribs[:, :len(X_df.columns)]
            return contribs
        else:
            # try sklearn wrapper predict with pred_contribs
            try:
                contribs = model.predict(X_np, pred_contribs=True)
                contribs = np.array(contribs)
                if contribs.ndim == 2 and contribs.shape[1] == len(X_df.columns) + 1:
                    contribs = contribs[:, :len(X_df.columns)]
                return contribs
            except Exception:
                return None
    except Exception:
        print("compute_xgb_contribs failed:", traceback.format_exc())
        return None

@app.on_event("startup")
def startup_event():
    global MODEL, MODEL_FEATURE_ORDER, EXPLAINER, LTV_LOW_THRESHOLD, LTV_MED_THRESHOLD, USE_CONTRIBS_FORCED

    try:
        MODEL, MODEL_FEATURE_ORDER = load_model_and_feature_order()
        print("Model loaded. Feature order:")
        _debug_print("MODEL_FEATURE_ORDER", MODEL_FEATURE_ORDER)
    except Exception:
        MODEL = None
        MODEL_FEATURE_ORDER = None
        print("Failed to load model at startup:", traceback.format_exc())

    LTV_LOW_THRESHOLD = safe_float(os.environ.get("LTV_LOW_THRESHOLD", LTV_LOW_THRESHOLD))
    LTV_MED_THRESHOLD = safe_float(os.environ.get("LTV_MED_THRESHOLD", LTV_MED_THRESHOLD))
    print(f"Thresholds: low={LTV_LOW_THRESHOLD}, med={LTV_MED_THRESHOLD}")

    # If forced toggle is set, skip building SHAP explainer
    EXPLAINER = None
    if USE_CONTRIBS_FORCED:
        print("FORCE_USE_CONTRIBS is true — skipping SHAP explainer build and using XGBoost contribs for explanations.")
        return

    # Attempt to build SHAP explainer
    if MODEL is None:
        print("Model not loaded; cannot build explainer.")
        return

    sample_X = None
    try:
        sample_X = pd.DataFrame([{f: 0.0 for f in MODEL_FEATURE_ORDER}])
    except Exception:
        sample_X = None

    print("Attempting to build SHAP explainer.")
    EXPLAINER = build_shap_explainer(MODEL, sample_X=sample_X)
    print("EXPLAINER built:", EXPLAINER is not None)

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": MODEL is not None,
        "explainer_loaded": EXPLAINER is not None and not USE_CONTRIBS_FORCED,
        "feature_count": len(MODEL_FEATURE_ORDER) if MODEL_FEATURE_ORDER else 0,
        "thresholds": {"low": LTV_LOW_THRESHOLD, "med": LTV_MED_THRESHOLD},
        "force_use_contribs": USE_CONTRIBS_FORCED
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

        _debug_print("Input shape", {"shape": X.shape, "columns": X.columns.tolist()})

        preds_raw = MODEL.predict(X)
        preds = [float(max(0.0, p)) for p in preds_raw]

        low_th = safe_float(os.environ.get("LTV_LOW_THRESHOLD", LTV_LOW_THRESHOLD))
        med_th = safe_float(os.environ.get("LTV_MED_THRESHOLD", LTV_MED_THRESHOLD))

        need_expl = bool(req.return_explanation)
        shap_values = None
        contribs = None

        if need_expl:
            # If forced contribs, skip SHAP and compute contribs directly
            if USE_CONTRIBS_FORCED:
                contribs = compute_xgb_contribs(MODEL, X)
                if contribs is None:
                    print("Forced contribs requested but compute_xgb_contribs returned None.")
            else:
                # Try SHAP explainer first
                if EXPLAINER is not None:
                    try:
                        expl_res = EXPLAINER(X)
                        shap_values = np.array(expl_res.values) if hasattr(expl_res, "values") else np.array(expl_res)
                        _debug_print("shap_values shape", getattr(shap_values, "shape", None))
                        if shap_values.ndim != 2 or shap_values.shape[1] != len(MODEL_FEATURE_ORDER):
                            print("Unexpected shap_values shape; will attempt contribs fallback.")
                            shap_values = None
                    except Exception:
                        print("SHAP runtime failed; will attempt contribs fallback.")
                        print(traceback.format_exc())
                        shap_values = None

                # If SHAP failed or not present, try contribs fallback
                if shap_values is None:
                    contribs = compute_xgb_contribs(MODEL, X)
                    if contribs is None:
                        print("compute_xgb_contribs returned None.")

        # Build response
        results: List[Dict[str, Any]] = []
        for i, cid in enumerate(ids):
            pred = preds[i]
            seg = ltv_to_segment(pred, low_th, med_th)
            item: Dict[str, Any] = {"customer_id": cid, "predicted_LTV": pred, "segment": seg}

            explanation_list = None
            if shap_values is not None:
                row = shap_values[i]
                abs_vals = np.abs(row)
                top_idx = abs_vals.argsort()[::-1][:3]
                explanation_list = [{"feature": MODEL_FEATURE_ORDER[idx], "impact": float(row[idx])} for idx in top_idx]
            elif contribs is not None:
                row = contribs[i]
                abs_vals = np.abs(row)
                top_idx = abs_vals.argsort()[::-1][:3]
                explanation_list = [{"feature": MODEL_FEATURE_ORDER[idx], "impact": float(row[idx])} for idx in top_idx]

            if explanation_list is not None:
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
