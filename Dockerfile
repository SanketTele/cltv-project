# Use official Python image with build tools
FROM python:3.10-slim

# Install system-level dependencies for SHAP
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    build-essential \
    libomp-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements into image
COPY src/requirements.txt /app/requirements.txt

# Install dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r /app/requirements.txt

# Copy code
COPY src /app/src
COPY models /app/models

# Expose API port
EXPOSE 8000

# Start FastAPI server
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
