# Dockerfile (placed at repo root: cltv-project/Dockerfile)
FROM python:3.10-slim

# Install system build tools needed for shap and xgboost wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    libomp-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements file and install dependencies
COPY src/requirements.txt /app/requirements.txt
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r /app/requirements.txt

# Copy application code and models
COPY src /app/src
COPY models /app/models

# Expose port 8000 for FastAPI
EXPOSE 8000

# Run the fastapi app with uvicorn
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
