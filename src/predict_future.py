
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "netflix_prepared.csv"
MODEL_PATH = BASE_DIR / "models" / "best_model.joblib"
OUTPUT_DIR = BASE_DIR / "outputs"
PLOTS_DIR = OUTPUT_DIR / "plots"


def make_features(dates, start_index):
    dates = pd.DatetimeIndex(dates)

    return pd.DataFrame({
        "time_index": np.arange(
            start_index, start_index + len(dates)
        ),
        "month_sin": np.sin(2 * np.pi * dates.month / 12),
        "month_cos": np.cos(2 * np.pi * dates.month / 12)
    })


def predict_future(forecast_years=5):
    if forecast_years < 1:
        raise ValueError("forecast_years must be at least 1.")

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "Prepared data not found. Run preprocessing first."
        )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Trained model not found. Run forecasting_models.py first."
        )

    df = pd.read_csv(DATA_PATH, parse_dates=["date_added"])

    # Use the same historical period as model training
    df = df[df["date_added"].dt.year.between(2015, 2020)].copy()

    monthly = (
        df.set_index("date_added")
        .resample("MS")
        .size()
        .asfreq("MS", fill_value=0)
    )

    artifact = joblib.load(MODEL_PATH)
    model_type = artifact["model_type"]

    # Generate future months after the last training month
    forecast_months = forecast_years * 12

    future_dates = pd.date_range(
        start=monthly.index[-1] + pd.offsets.MonthBegin(1),
        periods=forecast_months,
        freq="MS"
    )

    # Predict according to the selected model
    if model_type == "naive":
        predictions = np.repeat(
            artifact["last_count"], forecast_months
        )

    elif model_type == "seasonal_naive":
        last_12 = np.asarray(artifact["last_12"], dtype=float)

        predictions = np.tile(
            last_12,
            int(np.ceil(forecast_months / len(last_12)))
        )[:forecast_months]

    elif model_type == "linear":
        future_X = make_features(
            future_dates, len(monthly)
        )
        predictions = artifact["estimator"].predict(future_X)

    elif model_type == "holt_winters":
        predictions = artifact["estimator"].forecast(
            forecast_months
        ).to_numpy()

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    predictions = np.maximum(
        np.rint(predictions), 0
    ).astype(int)

    forecast_df = pd.DataFrame({
        "month": future_dates,
        "predicted_titles_added": predictions
    })

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # Save monthly predictions
    forecast_path = OUTPUT_DIR / "future_predictions.csv"
    forecast_df.to_csv(forecast_path, index=False)

    # Plot history and forecast
    plt.figure(figsize=(13, 6))

    plt.plot(
        monthly.index,
        monthly.values,
        label="Historical monthly additions"
    )

    plt.plot(
        forecast_df["month"],
        forecast_df["predicted_titles_added"],
        linestyle="--",
        label=f"Forecast: {artifact['best_model_name']}"
    )

    plt.axvline(
        monthly.index[-1],
        linestyle=":",
        label="Forecast starts"
    )

    plt.title("Netflix Monthly Content Addition Forecast")
    plt.xlabel("Month")
    plt.ylabel("Number of titles added")
    plt.legend()
    plt.tight_layout()

    chart_path = PLOTS_DIR / "forecast.png"
    plt.savefig(chart_path, dpi=150)
    plt.close()

    print("\nFUTURE MONTHLY FORECAST")
    print("=" * 55)
    print("Selected model:", artifact["best_model_name"])
    print("Forecast period:", future_dates[0], "to", future_dates[-1])
    print("\nFirst 12 forecast months:")
    print(forecast_df.head(12).to_string(index=False))
    print("\nSaved predictions:", forecast_path)
    print("Saved chart:", chart_path)

    return forecast_df


if __name__ == "__main__":
    predict_future(forecast_years=5)
