import requests, json
url = "https://cltv-api.onrender.com/predict"
payload = {
  "customers":[
    {
      "customer_id":"C101",
      "frequency":4,
      "total_spend":200,
      "aov":50,
      "recency_days":20,
      "T_days":300,
      "avg_interpurchase_days":100,
      "active_months":3,
      "purchase_days_std":12,
      "category_diversity":2,
      "avg_order_value":50,
      "unique_days":4
    }
  ],
  "return_explanation": True
}
r = requests.post(url, json=payload, timeout=30)
print("Status:", r.status_code)
print(json.dumps(r.json(), indent=2))
