"""
generate_sales_data.py
Simulates realistic daily multi-item retail transactions spanning 3 years (2023 - 2025).
Incorporates:
- Baseline trend growth
- Weekly seasonality (weekend sales boosts)
- Annual seasonality (holiday spikes in Q4, summer lifts)
- Realistic stochastic demand noise
"""

import os
import numpy as np
import pandas as pd

def generate_retail_sales():
    np.random.seed(42)

    # 3-year daily time index
    dates = pd.date_range(start="2023-01-01", end="2025-12-31", freq="D")
    n_days = len(dates)

    # Product categories with distinct baseline parameters
    products = {
        "SKU-101": {"name": "Wireless Audio", "base": 120, "trend": 0.08, "price": 49.99},
        "SKU-202": {"name": "Mechanical Keyboards", "base": 85, "trend": 0.05, "price": 89.99},
        "SKU-303": {"name": "USB-C Hub Adapters", "base": 210, "trend": 0.12, "price": 29.99},
    }

    records = []

    for sku, info in products.items():
        t = np.arange(n_days)

        # 1. Base + Long-term Trend
        trend_component = info["base"] + (info["trend"] * t)

        # 2. Day-of-week Seasonality (peaks on Fri/Sat/Sun)
        day_of_week = dates.dayofweek
        weekly_pattern = np.array([0.9, 0.95, 1.0, 1.05, 1.25, 1.35, 1.15])
        weekly_component = weekly_pattern[day_of_week]

        # 3. Yearly Seasonality (peaks in Nov/Dec holidays and July summer)
        day_of_year = dates.dayofyear
        annual_cycle = 1.0 + 0.20 * np.sin(2 * np.pi * (day_of_year - 80) / 365.25)

        # Q4 Holiday uplift (Black Friday & Christmas)
        q4_boost = np.where(
            ((dates.month == 11) & (dates.day >= 20)) | ((dates.month == 12) & (dates.day <= 26)),
            1.45,
            1.0
        )

        # 4. Stochastic Noise (Gaussian variation)
        noise = np.random.normal(loc=1.0, scale=0.08, size=n_days)

        # Calculate final daily units sold
        units_sold = trend_component * weekly_component * annual_cycle * q4_boost * noise
        units_sold = np.maximum(0, np.round(units_sold)).astype(int)

        revenue = np.round(units_sold * info["price"], 2)

        for i in range(n_days):
            records.append({
                "date": dates[i],
                "sku": sku,
                "product_name": info["name"],
                "unit_price": info["price"],
                "units_sold": units_sold[i],
                "revenue": revenue[i],
                "is_weekend": int(dates[i].dayofweek >= 5),
                "is_holiday_season": int(q4_boost[i] > 1.0),
            })

    df = pd.DataFrame(records)

    # Save outputs
    output_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, "sample_sales.csv")
    df.to_csv(csv_path, index=False)

    print(f"[SUCCESS] Generated {len(df):,} transaction rows.")
    print(f"[SAVED] File written to: {csv_path}")
    print("\nDataset Snapshot:")
    print(df.head(6))

if __name__ == "__main__":
    generate_retail_sales()