
from pathlib import Path
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "netflix_prepared.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
PLOTS_DIR = OUTPUT_DIR / "plots"


def analyze_trends():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "Prepared data not found. Run data_preprocessing.py first."
        )

    df = pd.read_csv(DATA_PATH, parse_dates=["date_added"])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # Annual content additions
    yearly = (
        df.groupby(["added_year", "type"])
        .size()
        .unstack(fill_value=0)
        .sort_index()
    )

    yearly.index.name = "year"
    yearly.to_csv(OUTPUT_DIR / "yearly_trends.csv")

    print("\nYEARLY CONTENT ADDITIONS")
    print(yearly.to_string())

    # Annual trend chart
    yearly.plot(
        kind="bar",
        figsize=(12, 6),
        width=0.8
    )
    plt.title("Netflix Content Added by Year and Type")
    plt.xlabel("Year")
    plt.ylabel("Number of Titles")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "yearly_trend.png", dpi=150)
    plt.close()

    # Monthly seasonality across the entire dataset
    monthly = (
        df.groupby("added_month")
        .size()
        .reindex(range(1, 13), fill_value=0)
    )

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 5))
    sns.barplot(x=monthly.index, y=monthly.values)
    plt.title("Netflix Content Additions by Month")
    plt.xlabel("Month")
    plt.ylabel("Number of Titles")
    plt.xticks(range(12), [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
    ])
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "monthly_trend.png", dpi=150)
    plt.close()

    # Overall annual totals for forecasting
    annual_total = (
        df.groupby("added_year")
        .size()
        .rename("title_count")
        .sort_index()
    )

    print("\nANNUAL TOTALS")
    print(annual_total.to_string())

    print(f"\nCharts saved in: {PLOTS_DIR}")
    print(f"Yearly data saved in: {OUTPUT_DIR}")

    return annual_total


if __name__ == "__main__":
    analyze_trends()
