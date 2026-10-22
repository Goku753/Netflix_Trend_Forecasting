
from src.data_preprocessing import prepare_data
from src.trend_analysis import analyze_trends
from src.forecasting_models import train_models
from src.evaluate_model import evaluate_model
from src.predict_future import predict_future


def main():
    print("=" * 60)
    print(" NETFLIX TREND FORECASTING MODEL")
    print("=" * 60)

    print("\nSTEP 1: Data preprocessing")
    prepare_data()

    print("\nSTEP 2: Historical trend analysis")
    analyze_trends()

    print("\nSTEP 3: Train and compare forecasting models")
    train_models()

    print("\nSTEP 4: Evaluate forecasting accuracy")
    evaluate_model()

    print("\nSTEP 5: Generate future predictions")
    predict_future(forecast_years=5)

    print("\n" + "=" * 60)
    print("PROJECT COMPLETED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    main()

