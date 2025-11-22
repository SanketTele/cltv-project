# main.py

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

# Create FastAPI app
app = FastAPI(
    title="Customer Lifetime Value (CLTV) Prediction API",
    description=(
        "Backend service for CLTV prediction with XGBoost, "
        "segmentation, and SHAP-based explanations."
    ),
    version="1.0.0",
)

# ---------------- Root / Homepage Route ---------------- #

@app.get("/", response_class=HTMLResponse, tags=["Root"])
async def homepage():
    return """
    <html>
        <head>
            <title>Customer Lifetime Value (CLTV) Prediction API</title>
        </head>
        <body style="font-family: Arial, sans-serif; max-width: 800px; margin: 40px auto;">
            <h1>Customer Lifetime Value (CLTV) Prediction API ✅</h1>
            <p><b>Status:</b> Running</p>
            <p>This backend powers the end-to-end CLTV prediction project with:</p>
            <ul>
                <li>Data ingestion & feature engineering</li>
                <li>XGBoost model training & deployment</li>
                <li>Threshold-based CLTV segmentation</li>
                <li>SHAP explanations (FORCE_USE_CONTRIBS)</li>
                <li>Streamlit frontend for interactive predictions</li>
            </ul>

            <h2>Available endpoints</h2>
            <ul>
                <li><code>/health</code> – Health check</li>
                <li><code>/docs</code> – API documentation</li>
                <li><code>/predict</code> – CLTV prediction + SHAP</li>
            </ul>

            <p>👉 <a href="/docs">Open API Docs</a></p>
        </body>
    </html>
    """

