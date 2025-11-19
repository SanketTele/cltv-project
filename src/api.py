# api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
import traceback
import os

# Import helper functions that load the model and prepare inputs
# Ensure src/model_utils.py exists and exports load_model_and_feature_order, prepare_input_df
from .model_utils import load_model_and_feature_order, prepare_input_df

app = FastAPI(title="CLTV Prediction API", version="1.0")

# ==========================
# Friendly root endpoint
# (Placed right after app = FastAPI(...) so decorator registers correctly)
# ==========================
@app.get("/")
def root():
    return {
        "service": "cltv-api",
        "status": "running",
        "description": "Customer Lifetime Value prediction API",
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "predict (POST)": "/predict"
        }
    }

# Global variables for model + feature order
MODEL = None
MODEL_FEATURE_ORDER = None

# ==========================
# REQUEST & RESPONSE MODELS
# ==========================
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


# ==========================
# STARTUP – Load Model & Feature Order
# ==========================
@app.on_event("startup")
def startup_event():
    global MODEL, MODEL_FEATURE_ORDER
    try:
        # load_model_and_feature_order() should return (model_object, feature_order_list)
        MODEL, MODEL_FEATURE_ORDER = load_model_and_feature_order()
        print("\n🚀 Model Loaded Successfully")
        print("📌 Feature Order:", MODEL_FEATURE_ORDER)
    except Exception as e:
        MODEL = None
        MODEL_FEATURE_ORDER = None
        print("\n❌ ERROR LOADING MODEL:", str(e))


# ==========================
# HEALTH CHECK
# ==========================
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": MODEL is not None,
        "feature_count": len(MODEL_FEATURE_ORDER) if MODEL_FEATURE_ORDER else 0
    }


# ==========================
# PREDICTION ENDPOINT
# ==========================
@app.post("/predict", response_model=List[PredictResponseItem])
def predict(req: PredictRequest):
    try:
        if MODEL is None or MODEL_FEATURE_ORDER is None:
            raise HTTPException(status_code=503, detail="Model not loaded on server.")

        customers_list = [c.dict() for c in req.customers]

        # prepare input dataframe in the exact order expected by model
        X, ids = prepare_input_df(customers_list, MODEL_FEATURE_ORDER)

        # get predictions
        preds = MODEL.predict(X)
        preds = [float(max(0, p)) for p in preds]  # clamp negatives to 0.0

        results = [{"customer_id": cid, "predicted_LTV": pred} for cid, pred in zip(ids, preds)]
        return results

    except HTTPException:
        raise
    except Exception:
        traceback_str = traceback.format_exc()
        raise HTTPException(status_code=500, detail=traceback_str)
