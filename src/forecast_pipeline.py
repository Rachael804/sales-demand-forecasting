"""
forecast_pipeline.py
End-to-end forecasting pipeline:
- Train/Test chronological split (last 60 days held out for validation)
- Fits a SARIMAX(1, 1, 1)x(1, 1, 1, 7) model with weekly seasonality
- Evaluates out-of-sample performance via MAE, RMSE, and MAPE
- Generates forecast vs. actual test plots
- Serializes trained model artifact to models/
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error

def train_and_evaluate(sku="SKU-101", test_days=60):
    # Load dataset
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "sample_sales.csv")
    df = pd.read_csv(data_path, parse_dates=["date"])
    
    sku_df = df[df["sku"] == sku].sort_values("date").set_index("date")
    series = sku_df["units_sold"].asfreq("D")

    # Chronological Train / Test Split
    train = series.iloc[:-test_days]
    test = series.iloc[-test_days:]

    print("==================================================")
    print(f" TRAINING SARIMAX FORECAST MODEL: {sku}")
    print(f" Train window: {train.index.min().date()} to {train.index.max().date()} ({len(train)} days)")
    print(f" Test window : {test.index.min().date()} to {test.index.max().date()} ({len(test)} days)")
    print("==================================================")

    # Fit SARIMAX: order=(1,1,1), seasonal_order=(1,1,1,7)
    model = SARIMAX(
        train,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 7),
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    results = model.fit(disp=False)
    print("\n[SUCCESS] Model training complete.")

    # Generate out-of-sample forecasts
    forecast_res = results.get_forecast(steps=test_days)
    predictions = forecast_res.predicted_mean
    conf_int = forecast_res.conf_int()

    # Evaluation Metrics
    mae = mean_absolute_error(test, predictions)
    rmse = np.sqrt(mean_squared_error(test, predictions))
    mape = np.mean(np.abs((test - predictions) / test)) * 100

    print("\nModel Evaluation Performance (Holdout Test Set):")
    print(f"  - MAE  (Mean Absolute Error)     : {mae:.2f} units")
    print(f"  - RMSE (Root Mean Squared Error) : {rmse:.2f} units")
    print(f"  - MAPE (Mean Abs Percentage Err) : {mape:.2f}%")

    # Save visualization
    figures_dir = os.path.join(os.path.dirname(__file__), "..", "reports", "figures")
    os.makedirs(figures_dir, exist_ok=True)
    plot_path = os.path.join(figures_dir, f"{sku.lower()}_forecast_eval.png")

    plt.figure(figsize=(13, 6))
    plt.plot(train.index[-90:], train.iloc[-90:], label="Recent Training Actuals", color="#334155")
    plt.plot(test.index, test, label="Actual Demand (Holdout)", color="#0f172a", linewidth=2)
    plt.plot(test.index, predictions, label="SARIMAX Forecast", color="#0d9488", linewidth=2)
    plt.fill_between(
        test.index,
        conf_int.iloc[:, 0],
        conf_int.iloc[:, 1],
        color="#2dd4bf",
        alpha=0.25,
        label="95% Confidence Interval"
    )
    plt.title(f"Sales Demand Forecast vs. Ground Truth - {sku} (SARIMAX)", fontsize=13)
    plt.xlabel("Date")
    plt.ylabel("Units Sold")
    plt.legend(loc="upper left")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"\n[SAVED] Forecast plot saved to: {plot_path}")

    # Save model artifact
    models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, f"{sku.lower()}_sarimax.pkl")
    joblib.dump(results, model_path)
    print(f"[SAVED] Serialized model saved to: {model_path}")

if __name__ == "__main__":
    train_and_evaluate()