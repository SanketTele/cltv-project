# api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
import traceback

# Import helper functions
from .model_utils import load_model_and_feature_order, prepare_input_df

app = FastAPI(title="CLTV Prediction API", version="1.0")

# Global storage
MODEL = None
MODEL_FEATURE_ORDER = None


# ============================================================
# 1. REQUEST & RESPONSE MODELS (defined before usage)
# ============================================================

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


class PredictRequest(BaseModel):
    customers: List[CustomerFeatures]


class PredictResponseItem(BaseModel):
    customer_id: str
    predicted_LTV: float


# ============================================================
# 2. STARTUP – Load Model & Feature Order
# ============================================================

@app.on_event("startup")
def startup_event():
    global MODEL, MODEL_FEATURE_ORDER
    try:
        MODEL, MODEL_FEATURE_ORDER = load_model_and_feature_order()
        print("\n🚀 Model Loaded Successfully")
        print("📌 Feature Order:", MODEL_FEATURE_ORDER)
    except Exception as e:
        print("\n❌ ERROR LOADING MODEL:", str(e))
        MODEL = None
        MODEL_FEATURE_ORDER = None


# ============================================================
# 3. HEALTH CHECK
# ============================================================

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": MODEL is not None,
        "feature_count": len(MODEL_FEATURE_ORDER) if MODEL_FEATURE_ORDER else 0
    }


# ============================================================
# 4. PREDICTION ENDPOINT
# ============================================================

@app.post("/predict", response_model=List[PredictResponseItem])
def predict(req: PredictRequest):
    try:
        if MODEL is None or MODEL_FEATURE_ORDER is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        # Convert JSON -> DataFrame
        customers_list = [c.dict() for c in req.customers]
        X, ids = prepare_input_df(customers_list, MODEL_FEATURE_ORDER)

        # Predict
        preds = MODEL.predict(X)
        preds = [float(max(0, p)) for p in preds]

        # Build response
        output = [
            {"customer_id": cid, "predicted_LTV": pred}
            for cid, pred in zip(ids, preds)
        ]

        return output

    except Exception as e:
        traceback_str = traceback.format_exc()
        raise HTTPException(status_code=500, detail=traceback_str)
