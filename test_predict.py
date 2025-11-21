import requests
import json

# Your API endpoint
url = "https://cltv-api.onrender.com/predict"

# Request payload for prediction only
payload = {
    "customers": [
        {
            "customer_id": "C101",
            "frequency": 4,
            "total_spend": 200,
            "aov": 50,
            "recency_days": 20,
            "T_days": 300,
            "avg_interpurchase_days": 100,
            "active_months": 3,
            "purchase_days_std": 12,
            "category_diversity": 2,
            "avg_order_value": 50,
            "unique_days": 4
        }
    ],
    "return_explanation": False
}

try:
    response = requests.post(url, json=payload, timeout=60)
    print("Status:", response.status_code)
    print(json.dumps(response.json(), indent=2))

except requests.exceptions.Timeout:
    print("⛔ ERROR: Request timed out. Try increasing timeout or check server load.")
except Exception as e:
    print("Unexpected error:", e)
