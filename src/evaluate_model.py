
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_PATH = BASE_DIR / "outputs" / "model_evaluation.csv"


def evaluate_model():
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(
            "Evaluation results not found. "
            "Run forecasting_models.py first."
        )

    results = pd.read_csv(RESULTS_PATH)
    results = results.sort_values("MAE").reset_index(drop=True)

    print("\nFORECASTING MODEL EVALUATION")
    print("=" * 55)
    print(results.to_string(index=False))

    best = results.iloc[0]

    print("\nBEST MODEL")
    print(f"Model: {best['model']}")
    print(f"MAE: {best['MAE']:.2f}")
    print(f"RMSE: {best['RMSE']:.2f}")

    if pd.notna(best["MAPE_percent"]):
        print(f"MAPE: {best['MAPE_percent']:.2f}%")
    else:
        print("MAPE: Not available")

    print("\nLower MAE and RMSE indicate better performance.")
    print("Compare against the naive baseline before choosing a model.")

    return results


if __name__ == "__main__":
    evaluate_model()
