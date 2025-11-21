# src/api.py
"""
Safe CLTV API that exposes `app` for Uvicorn.
Uses XGBoost pred_contribs as explanation fallback if SHAP is unavailable.
"""

import os, traceback
from typing import List, Optional, Dict, Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# optional SHAP
try:
    import shap
except Exception:
    shap = None

# optional XGBoost
try:
    import xgboost as xgb
except Exception:
    xgb = None

from .model_utils import load_model_and_feature_order, prepare_input_df

# === Expose app for Uvicorn ===
app = FastAPI(title="CLTV Prediction API", version="1.0")

# Globals
MODEL = None
MODEL_FEATURE_ORDER: Optional[List[str]] = None
EXPLAINER = None
USE_CONTRIBS_FORCED = os.environ.get("FORCE_USE_CONTRIBS", "false").lower() in ("1","true","yes")

LTV_LOW_THRESHOLD = float(os.environ.get("LTV_LOW_THRESHOLD", 50.0))
LTV_MED_THRESHOLD = float(os.environ.get("LTV_MED_THRESHOLD", 200.0))

# Schemas
class CustomerFeatures(BaseModel):
    customer_id: str
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

def ltv_to_segment(v, low, med):
    if v < low:
        return "Low"
    if v < med:
        return "Medium"
    return "High"

def compute_xgb_contribs(model, Xdf):
    try:
        X_np = Xdf.values if isinstance(Xdf, pd.DataFrame) else np.array(Xdf)
        booster = None
        if hasattr(model, "get_booster"):
            booster = model.get_booster()
        elif hasattr(model, "booster_"):
            booster = model.booster_
        if booster is not None and xgb is not None:
            dmat = xgb.DMatrix(X_np, feature_names=list(Xdf.columns) if hasattr(Xdf, "columns") else None)
            contribs = booster.predict(dmat, pred_contribs=True)
            contribs = np.array(contribs)
            if contribs.ndim == 2 and contribs.shape[1] == (Xdf.shape[1] + 1):
                contribs = contribs[:, : Xdf.shape[1]]
            return contribs
        else:
            try:
                contribs = model.predict(X_np, pred_contribs=True)
                contribs = np.array(contribs)
                if contribs.ndim == 2 and contribs.shape[1] == (Xdf.shape[1] + 1):
                    contribs = contribs[:, : Xdf.shape[1]]
                return contribs
            except Exception:
                return None
    except Exception:
        print("compute_xgb_contribs error:", traceback.format_exc())
        return None

@app.on_event("startup")
def startup_event():
    global MODEL, MODEL_FEATURE_ORDER, EXPLAINER
    try:
        MODEL, MODEL_FEATURE_ORDER = load_model_and_feature_order()
        print("Model loaded; features:", MODEL_FEATURE_ORDER)
    except Exception:
        MODEL = None
        MODEL_FEATURE_ORDER = None
        print("Model load failed:", traceback.format_exc())

    if USE_CONTRIBS_FORCED:
        print("FORCE_USE_CONTRIBS set -> skipping SHAP")
        return

    if shap is not None and MODEL is not None:
        try:
            booster = None
            if hasattr(MODEL, "get_booster"): booster = MODEL.get_booster()
            elif hasattr(MODEL, "booster_"): booster = MODEL.booster_
            EXPLAINER = shap.TreeExplainer(booster if booster is not None else MODEL)
            print("SHAP explainer built")
        except Exception:
            EXPLAINER = None
            print("SHAP explainer build failed:", traceback.format_exc())

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": MODEL is not None,
        "explainer_loaded": EXPLAINER is not None and not USE_CONTRIBS_FORCED,
        "feature_count": len(MODEL_FEATURE_ORDER) if MODEL_FEATURE_ORDER else 0,
        "force_use_contribs": USE_CONTRIBS_FORCED,
        "thresholds": {"low": LTV_LOW_THRESHOLD, "med": LTV_MED_THRESHOLD}
    }

@app.post("/predict", response_model=List[PredictResponseItem])
def predict(req: PredictRequest):
    if MODEL is None or MODEL_FEATURE_ORDER is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    X, ids = prepare_input_df([c.dict() for c in req.customers], MODEL_FEATURE_ORDER)

    preds = MODEL.predict(X)
    preds = [float(max(0.0, p)) for p in preds]

    shap_values = None
    contribs = None

    if req.return_explanation:
        if not USE_CONTRIBS_FORCED and EXPLAINER is not None:
            try:
                expl = EXPLAINER(X)
                shap_values = np.array(expl.values) if hasattr(expl, "values") else np.array(expl)
            except Exception:
                print("SHAP runtime error:", traceback.format_exc())
                shap_values = None
        if shap_values is None:
            contribs = compute_xgb_contribs(MODEL, X)

    results = []
    for i, cid in enumerate(ids):
        res = {"customer_id": cid, "predicted_LTV": preds[i], "segment": ltv_to_segment(preds[i], LTV_LOW_THRESHOLD, LTV_MED_THRESHOLD)}
        expl = None
        if shap_values is not None:
            row = shap_values[i]
            top = np.abs(row).argsort()[::-1][:3]
            expl = [{"feature": MODEL_FEATURE_ORDER[j], "impact": float(row[j])} for j in top]
        elif contribs is not None:
            row = contribs[i]
            top = np.abs(row).argsort()[::-1][:3]
            expl = [{"feature": MODEL_FEATURE_ORDER[j], "impact": float(row[j])} for j in top]
        if expl is not None:
            res["explanation"] = expl
        results.append(res)

    return results
