# src/api.py
"""
CLTV Prediction API with forced contribs toggle.
Set environment variable FORCE_USE_CONTRIBS=true to always use pred_contribs fallback.
"""

import os
import traceback
from typing import List, Optional, Dict, Any

import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Try importing shap
try:
    import shap
except Exception:
    shap = None

# Try importing xgboost
try:
    import xgboost as xgb
except Exception:
    xgb = None

from .model_utils import load_model_and_feature_order, prepare_input_df

app = FastAPI(title="CLTV Prediction API", version="1.0")

MODEL = None
MODEL_FEATURE_ORDER: Optional[List[str]] = None
EXPLAINER = None

USE_CONTRIBS_FORCED = os.environ.get("FORCE_USE_CONTRIBS", "false").lower() in ("1", "true", "yes")

LTV_LOW_THRESHOLD = 50.0
LTV_MED_THRESHOLD = 200.0


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


def safe_float(x, default=0.0):
    try:
        return float(x)
    except:
        return default


def ltv_to_segment(l, low, med):
    if l < low:
        return "Low"
    if l < med:
        return "Medium"
    return "High"


def compute_xgb_contribs(model, Xdf):
    try:
        booster = model.get_booster()
        mat = xgb.DMatrix(Xdf.values, feature_names=list(Xdf.columns))
        contrib = booster.predict(mat, pred_contribs=True)
        contrib = np.array(contrib)[:, :-1]  # drop bias
        return contrib
    except:
        return None


@app.on_event("startup")
def startup_event():
    global MODEL, MODEL_FEATURE_ORDER, EXPLAINER, LTV_LOW_THRESHOLD, LTV_MED_THRESHOLD

    MODEL, MODEL_FEATURE_ORDER = load_model_and_feature_order()
    print("Model loaded")

    LTV_LOW_THRESHOLD = safe_float(os.environ.get("LTV_LOW_THRESHOLD", LTV_LOW_THRESHOLD))
    LTV_MED_THRESHOLD = safe_float(os.environ.get("LTV_MED_THRESHOLD", LTV_MED_THRESHOLD))

    if USE_CONTRIBS_FORCED:
        print("FORCE_USE_CONTRIBS = true → skipping SHAP explainer")
        return

    if shap is not None:
        try:
            EXPLAINER = shap.TreeExplainer(MODEL)
            print("SHAP explainer built")
        except:
            EXPLAINER = None
            print("Failed SHAP explainer build")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": MODEL is not None,
        "explainer_loaded": EXPLAINER is not None,
        "feature_count": len(MODEL_FEATURE_ORDER) if MODEL_FEATURE_ORDER else 0,
        "force_use_contribs": USE_CONTRIBS_FORCED
    }


@app.post("/predict", response_model=List[PredictResponseItem])
def predict(req: PredictRequest):

    if MODEL is None:
        raise HTTPException(503, "Model not loaded")

    X, ids = prepare_input_df([c.dict() for c in req.customers], MODEL_FEATURE_ORDER)

    preds_raw = MODEL.predict(X)
    preds = [float(max(0, p)) for p in preds_raw]

    low = LTV_LOW_THRESHOLD
    med = LTV_MED_THRESHOLD

    results = []

    # Explanation
    contribs = compute_xgb_contribs(MODEL, X) if req.return_explanation else None

    for i, cid in enumerate(ids):
        item = {
            "customer_id": cid,
            "predicted_LTV": preds[i],
            "segment": ltv_to_segment(preds[i], low, med)
        }

        if contribs is not None:
            row = contribs[i]
            abs_vals = np.abs(row)
            top_idx = abs_vals.argsort()[::-1][:3]

            item["explanation"] = [
                {"feature": MODEL_FEATURE_ORDER[j], "impact": float(row[j])}
                for j in top_idx
            ]

        results.append(item)

    return results
