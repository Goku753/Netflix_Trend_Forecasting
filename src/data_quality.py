
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "netflix_prepared.csv"


def check_data_quality():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "Prepared dataset not found. Run python main.py first."
        )

    df = pd.read_csv(DATA_PATH, parse_dates=["date_added"])

    print("\n" + "=" * 55)
    print("NETFLIX DATA QUALITY REPORT")
    print("=" * 55)

    print(f"\nTotal records: {len(df)}")
    print(f"Date range: {df['date_added'].min()} to "
          f"{df['date_added'].max()}")

    # Monthly coverage
    monthly = (
        df.groupby(df["date_added"].dt.to_period("M"))
        .size()
        .rename("title_count")
        .reset_index()
    )
    monthly["month"] = monthly["date_added"].astype(str)

    print("\nFirst 12 months in the dataset:")
    print(monthly[["month", "title_count"]].head(12).to_string(index=False))

    # Annual counts by content type
    annual = (
        df.groupby(["added_year", "type"])
        .size()
        .unstack(fill_value=0)
        .sort_index()
    )

    print("\nAnnual additions by content type:")
    print(annual.to_string())

    # Check whether each year has all 12 calendar months
    monthly_coverage = df.groupby("added_year")["date_added"].agg(
        first_date="min",
        last_date="max",
        distinct_months=lambda dates: dates.dt.month.nunique()
    )

    print("\nAnnual date coverage:")
    print(monthly_coverage.to_string())

    # Save the report tables
    output_dir = BASE_DIR / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    annual.to_csv(output_dir / "yearly_trends.csv")
    monthly.to_csv(output_dir / "monthly_counts.csv", index=False)
    monthly_coverage.to_csv(output_dir / "year_coverage.csv")

    print(f"\nReports saved to: {output_dir}")

    return annual, monthly, monthly_coverage


if __name__ == "__main__":
    check_data_quality()
