"""
time_series_analysis.py
Performs statistical diagnostics on daily sales demand:
- Augmented Dickey-Fuller (ADF) test for stationarity
- Classical additive seasonal decomposition (Trend, Seasonal, Residual)
- Generates publication-ready figures for documentation
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose

def run_diagnostics(sku="SKU-101"):
    # Load dataset
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "sample_sales.csv")
    df = pd.read_csv(data_path, parse_dates=["date"])
    
    # Filter for target product and set regular datetime index
    sku_df = df[df["sku"] == sku].sort_values("date").set_index("date")
    daily_units = sku_df["units_sold"].asfreq("D")

    print(f"==================================================")
    print(f" TIME-SERIES DIAGNOSTICS: {sku} ({sku_df['product_name'].iloc[0]})")
    print(f"==================================================")

    # 1. Augmented Dickey-Fuller (ADF) Stationarity Test
    adf_result = adfuller(daily_units.dropna())
    print("\n[1] Augmented Dickey-Fuller (ADF) Test:")
    print(f"  - ADF Statistic      : {adf_result[0]:.4f}")
    print(f"  - p-value            : {adf_result[1]:.4e}")
    print(f"  - Critical Value (1%): {adf_result[4]['1%']:.4f}")
    print(f"  - Critical Value (5%): {adf_result[4]['5%']:.4f}")
    
    if adf_result[1] < 0.05:
        print("  -> Result: Stationarity confirmed (p < 0.05). Reject H0.")
    else:
        print("  -> Result: Non-stationary (p >= 0.05). Differencing (d=1) required.")

    # 2. Seasonal Decomposition (Weekly period = 7 days)
    decomposition = seasonal_decompose(daily_units, model="additive", period=7)

    # 3. Save Decomposition Plot
    fig = decomposition.plot()
    fig.set_size_inches(12, 8)
    fig.suptitle(f"Seasonal Decomposition (7-Day Period) - {sku}", fontsize=14, y=1.02)
    
    figures_dir = os.path.join(os.path.dirname(__file__), "..", "reports", "figures")
    os.makedirs(figures_dir, exist_ok=True)
    plot_path = os.path.join(figures_dir, f"{sku.lower()}_decomposition.png")
    
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"\n[2] Diagnostic Chart Saved:")
    print(f"  -> Path: {plot_path}")

if __name__ == "__main__":
    run_diagnostics()