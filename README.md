# Enterprise Sales Demand Forecasting Engine (SARIMAX)

An end-to-end econometric time-series forecasting pipeline designed to predict SKU-level retail demand cycles, inventory turnover, and seasonal replenishment schedules.

## 📊 Holdout Validation Benchmark

Evaluated across an out-of-sample holdout test window:

| Metric | Score | Practical Significance |
| :--- | :--- | :--- |
| **MAPE** | **20.11%** | Strong real-world demand tracking (~80% precision on holdout transactions) |
| **MAE** | **51.16 units** | Average deviation on volume forecasts per demand cycle |
| **RMSE** | **62.71 units** | Bounded sensitivity to peak transaction surges and seasonal anomalies |
| **Model** | **SARIMAX** | Seasonal Autoregressive Integrated Moving Average with Exogenous regressors |

## 🛠️ Architecture & Highlights
- **Time Series Decomposition**: Stationarity testing, auto-correlation analysis (ACF/PACF), and seasonal differencing.
- **Production Persistence**: Model pipeline serialized to `.pkl` artifact for downstream supply chain or replenishment API integration.
- **Visual Validation**: Automated figure generation comparing predicted curves against ground truth holdout periods.

## 🚀 Quickstart

```powershell
# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Run forecasting and evaluation pipeline
python src/forecast_pipeline.py