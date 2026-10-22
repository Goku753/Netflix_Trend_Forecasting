
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from statsmodels.tsa.holtwinters import ExponentialSmoothing


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "netflix_prepared.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_DIR = BASE_DIR / "models"


def calculate_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))

    nonzero = actual != 0
    if nonzero.any():
        mape = np.mean(
            np.abs(
                (actual[nonzero] - predicted[nonzero])
                / actual[nonzero]
            )
        ) * 100
    else:
        mape = np.nan

    return mae, rmse, mape


def make_features(dates):
    dates = pd.DatetimeIndex(dates)

    return pd.DataFrame({
        "time_index": np.arange(len(dates)),
        "month_sin": np.sin(2 * np.pi * dates.month / 12),
        "month_cos": np.cos(2 * np.pi * dates.month / 12)
    })


def get_predictions(model_name, train, test):
    if model_name == "Naive Baseline":
        return np.repeat(train.iloc[-1], len(test))

    if model_name == "Seasonal Naive":
        previous_year = train.iloc[-12:].to_numpy()
        return np.tile(
            previous_year,
            int(np.ceil(len(test) / 12))
        )[:len(test)]

    if model_name == "Linear Regression":
        model = make_pipeline(
            StandardScaler(),
            LinearRegression()
        )
        model.fit(make_features(train.index), train.to_numpy())
        return np.maximum(
            model.predict(make_features(test.index)), 0
        )

    if model_name == "Holt-Winters":
        model = ExponentialSmoothing(
            train,
            trend="add",
            seasonal="add",
            seasonal_periods=12,
            initialization_method="estimated"
        ).fit(optimized=True)

        return np.maximum(
            model.forecast(len(test)).to_numpy(), 0
        )

    raise ValueError(f"Unknown model: {model_name}")


def train_models():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "Prepared dataset not found. Run preprocessing first."
        )

    df = pd.read_csv(DATA_PATH, parse_dates=["date_added"])

    # Keep the same six-year period as the previous experiment.
    df = df[
        df["date_added"].dt.year.between(2015, 2020)
    ].copy()

    if df.empty:
        raise ValueError("No records found for years 2015–2020.")

    monthly = (
        df.set_index("date_added")
        .resample("MS")
        .size()
        .asfreq("MS", fill_value=0)
    )

    if len(monthly) != 72:
        raise ValueError(
            f"Expected 72 months for 2015–2020, found {len(monthly)}."
        )

    model_names = [
        "Naive Baseline",
        "Seasonal Naive",
        "Linear Regression",
        "Holt-Winters"
    ]

    # Test three separate 12-month periods: 2018, 2019, and 2020.
    test_window_starts = [36, 48, 60]
    fold_results = []

    for start in test_window_starts:
        train = monthly.iloc[:start]
        test = monthly.iloc[start:start + 12]

        for model_name in model_names:
            predictions = get_predictions(model_name, train, test)
            mae, rmse, mape = calculate_metrics(test, predictions)

            fold_results.append({
                "model": model_name,
                "test_year": int(test.index[0].year),
                "MAE": mae,
                "RMSE": rmse,
                "MAPE_percent": mape
            })

    folds_df = pd.DataFrame(fold_results)

    # Average each model's scores across the three test years.
    results_df = (
        folds_df.groupby("model", as_index=False)
        .agg(
            MAE=("MAE", "mean"),
            RMSE=("RMSE", "mean"),
            MAPE_percent=("MAPE_percent", "mean")
        )
        .sort_values("MAE")
        .reset_index(drop=True)
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Save both the detailed fold scores and the summary.
    folds_df.to_csv(
        OUTPUT_DIR / "rolling_validation_results.csv",
        index=False
    )

    results_df.to_csv(
        OUTPUT_DIR / "model_evaluation.csv",
        index=False
    )

    # Select the model with the lowest average MAE.
    best_name = results_df.iloc[0]["model"]

    # Refit the winning model using all 72 months.
    if best_name == "Naive Baseline":
        artifact = {
            "model_type": "naive",
            "last_count": float(monthly.iloc[-1]),
            "frequency": "monthly",
            "best_model_name": best_name
        }

    elif best_name == "Seasonal Naive":
        artifact = {
            "model_type": "seasonal_naive",
            "last_12": monthly.iloc[-12:].to_numpy(),
            "frequency": "monthly",
            "best_model_name": best_name
        }

    elif best_name == "Linear Regression":
        final_model = make_pipeline(
            StandardScaler(),
            LinearRegression()
        )
        final_model.fit(
            make_features(monthly.index),
            monthly.to_numpy()
        )

        artifact = {
            "model_type": "linear",
            "estimator": final_model,
            "history_length": len(monthly),
            "frequency": "monthly",
            "best_model_name": best_name
        }

    else:
        final_model = ExponentialSmoothing(
            monthly,
            trend="add",
            seasonal="add",
            seasonal_periods=12,
            initialization_method="estimated"
        ).fit(optimized=True)

        artifact = {
            "model_type": "holt_winters",
            "estimator": final_model,
            "frequency": "monthly",
            "best_model_name": best_name
        }

    joblib.dump(artifact, MODEL_DIR / "best_model.joblib")

    print("\nROLLING-ORIGIN VALIDATION RESULTS")
    print("=" * 75)
    print("\nAverage scores across 2018, 2019, and 2020:")
    print(results_df.to_string(index=False))

    print("\nDetailed scores for each test year:")
    print(
        folds_df.sort_values(["test_year", "model"]).to_string(index=False)
    )

    print("\nBest model by average MAE:", best_name)
    print("Saved:", OUTPUT_DIR / "model_evaluation.csv")
    print("Saved:", OUTPUT_DIR / "rolling_validation_results.csv")
    print("Saved:", MODEL_DIR / "best_model.joblib")

    return results_df


if __name__ == "__main__":
    train_models()
