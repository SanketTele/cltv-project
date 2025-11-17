import os
import pandas as pd
from datetime import timedelta

RAW_CSV_PATH = r"C:\Users\Sanket\Desktop\Documents\Tasks\cltv-project\data\raw\online_retail_II.csv"
INTERIM_DIR = r"C:\Users\Sanket\Desktop\Documents\Tasks\cltv-project\data\interim"
OUTPUT_CSV = os.path.join(INTERIM_DIR, "transactions_clean.csv")

os.makedirs(INTERIM_DIR, exist_ok=True)

def main():

    print("Loading raw CSV...")
    df = pd.read_csv(RAW_CSV_PATH, encoding="latin1")

    print("Cleaning data...")
    df = df.rename(columns={
        "Invoice": "invoice_no",
        "InvoiceDate": "invoice_date",
        "Customer ID": "customer_id",
        "Quantity": "quantity",
        "Price": "unit_price",
        "Amount": "order_value"
    })

    # Convert date
    df["invoice_date"] = pd.to_datetime(df["invoice_date"], errors='coerce')

    # Remove missing values
    df = df.dropna(subset=["customer_id", "invoice_date"])

    # Convert numeric columns
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
    df["unit_price"] = pd.to_numeric(df["unit_price"], errors="coerce")

    # Create order_value if missing
    if "order_value" not in df.columns:
        df["order_value"] = df["quantity"] * df["unit_price"]

    # Remove zero or negative orders
    df = df[df["order_value"] > 0]

    # Save cleaned file
    df.to_csv(OUTPUT_CSV, index=False)

    print("Saved cleaned data at:", OUTPUT_CSV)
    print("Rows:", len(df))
    print("Unique customers:", df["customer_id"].nunique())


if __name__ == "__main__":
    main()
