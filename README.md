# Sales Demand Forecasting Engine

An end-to-end time-series modeling pipeline built in Python to forecast multi-item retail sales demand. The system evaluates stationarity, decomposes seasonal cycles, fits SARIMAX state-space models with weekly seasonality parameters, and benchmarks out-of-sample forecast accuracy on holdout test periods.

---

## Overview & Architecture

Accurate inventory planning requires modeling long-term sales trajectories alongside high-frequency demand shifts. This engine provides:

- **Synthetic Retail Data Generation:** Simulates multi-year daily transaction logs across distinct product SKUs incorporating baseline trend growth, 7-day cyclical patterns, Q4 holiday demand lifts, and stochastic noise.
- **Stationarity Diagnostics:** Automated Augmented Dickey-Fuller (ADF) testing to evaluate unit roots and verify differencing orders ($d=1$).
- **Classical Decomposition:** Additive component isolation across trend, seasonal, and residual components.
- **SARIMAX Time-Series Forecasting:** State-space modeling with seasonal order $(P, D, Q)_s = (1, 1, 1)_7$ capturing weekly reorder cycles.
- **Validation & Metrics:** 60-day out-of-sample holdout benchmarking measuring MAE, RMSE, and MAPE.

---

## Project Structure

```text
sales-demand-forecasting/
├── data/
│   ├── generate_sales_data.py       # Multi-SKU daily sales simulation script
│   └── sample_sales.csv             # Generated 3-year transactions dataset
├── src/
│   ├── time_series_analysis.py      # Stationarity testing & seasonal decomposition
│   └── forecast_pipeline.py         # Model training, holdout evaluation & artifact export
├── models/
│   └── sku-101_sarimax.pkl          # Serialized SARIMAX model artifact
├── reports/
│   └── figures/                     # Diagnostic & evaluation plots
│       ├── sku-101_decomposition.png
│       └── sku-101_forecast_eval.png
├── requirements.txt
├── .gitignore
└── README.md