
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "Dataset (1).csv"
OUTPUT_PATH = BASE_DIR / "data" / "netflix_prepared.csv"


def prepare_data():
    # Check whether the original dataset exists
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
        )

    # Load the dataset
    df = pd.read_csv(DATASET_PATH)

    print("Original dataset shape:", df.shape)

    # Validate required columns
    required_columns = {"title", "type", "date_added"}
    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Convert date_added to datetime
    df["date_added"] = pd.to_datetime(
        df["date_added"].astype("string").str.strip(),
        format="mixed",
        errors="coerce"
    )

    invalid_dates = df["date_added"].isna().sum()

    # Remove records without valid dates
    df = df.dropna(subset=["date_added"]).copy()

    # Create time-based features
    df["added_year"] = df["date_added"].dt.year
    df["added_month"] = df["date_added"].dt.month
    df["added_quarter"] = df["date_added"].dt.quarter
    df["added_month_name"] = df["date_added"].dt.month_name()
    df["added_day"] = df["date_added"].dt.day
    df["added_day_of_week"] = df["date_added"].dt.dayofweek

    # Create output directory and save processed data
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print("\nData preprocessing completed successfully!")
    print("Invalid dates:", invalid_dates)
    print("Valid records:", len(df))
    print("Date range:", df["date_added"].min(), "to",
          df["date_added"].max())
    print("Saved file:", OUTPUT_PATH)

    return df


if __name__ == "__main__":
    prepare_data()
