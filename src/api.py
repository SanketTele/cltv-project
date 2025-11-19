# api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import traceback
import os

# helper functions
from .model_utils import load_model_and_feature_order, prepare_input_df

app = FastAPI(title="CLTV Prediction API", version="1.0")

# Friendly root
@app.get("/")
def root():
    return {
        "service": "cltv-api",
        "status": "running",
        "description": "Customer Lifetime Value prediction API",
        "endpoints": {"health": "/health", "docs": "/docs", "predict (POST)": "/predict"}
    }

# Globals
MODEL = None
MODEL_FEATURE_ORDER = None

# Pydantic models
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

class PredictResponseItem(BaseModel):
    customer_id: str
    predicted_LTV: float
    segment: Optional[str] = None

# Utility: segmentation mapping
def ltv_to_segment(ltv: float, low_threshold: float, med_threshold: float) -> str:
    if ltv < low_threshold:
        return "Low"
    if ltv < med_threshold:
        return "Medium"
    return "High"

# Startup: load model and feature order
@app.on_event("startup")
def startup_event():
    global MODEL, MODEL_FEATURE_ORDER
    try:
        MODEL, MODEL_FEATURE_ORDER = load_model_and_feature_order()
        print("Model Loaded. Feature Order:", MODEL_FEATURE_ORDER)
        # read thresholds from environment (strings -> float) or use defaults
        global LTV_LOW_THRESHOLD, LTV_MED_THRESHOLD
        LTV_LOW_THRESHOLD = float(os.environ.get("LTV_LOW_THRESHOLD", "50"))
        LTV_MED_THRESHOLD = float(os.environ.get("LTV_MED_THRESHOLD", "200"))
        print(f"Segmentation thresholds: low={LTV_LOW_THRESHOLD}, med={LTV_MED_THRESHOLD}")
    except Exception as e:
        MODEL = None
        MODEL_FEATURE_ORDER = None
        print("ERROR LOADING MODEL:", str(e))

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": MODEL is not None, "feature_count": len(MODEL_FEATURE_ORDER) if MODEL_FEATURE_ORDER else 0}

@app.post("/predict", response_model=List[PredictResponseItem])
def predict(req: PredictRequest):
    try:
        if MODEL is None or MODEL_FEATURE_ORDER is None:
            raise HTTPException(status_code=503, detail="Model not loaded on server.")

        customers_list = [c.dict() for c in req.customers]
        X, ids = prepare_input_df(customers_list, MODEL_FEATURE_ORDER)
        preds = MODEL.predict(X)
        preds = [float(max(0, p)) for p in preds]  # no negative LTV

        # Use thresholds set in startup_event
        low_th = float(os.environ.get("LTV_LOW_THRESHOLD", LTV_LOW_THRESHOLD if 'LTV_LOW_THRESHOLD' in globals() else 50))
        med_th = float(os.environ.get("LTV_MED_THRESHOLD", LTV_MED_THRESHOLD if 'LTV_MED_THRESHOLD' in globals() else 200))

        results = []
        for cid, p in zip(ids, preds):
            seg = ltv_to_segment(p, low_th, med_th)
            results.append({"customer_id": cid, "predicted_LTV": p, "segment": seg})

        return results

    except HTTPException:
        raise
    except Exception:
        traceback_str = traceback.format_exc()
        raise HTTPException(status_code=500, detail=traceback_str)
