
# 🎬 Netflix Trend Forecasting & Analytics Studio

An interactive Data Science and Machine Learning project that analyzes historical Netflix catalog additions, compares time-series forecasting models, and generates monthly projections through 2025.

The project includes a professional Streamlit dashboard for exploring historical trends, evaluating model performance, visualizing forecasts, and downloading prediction results.

---

## 📌 Project Overview

Netflix's catalog changes over time as movies and TV shows are added. Understanding historical catalog patterns can help demonstrate how time-series analysis and forecasting techniques work.

This project uses the Netflix titles dataset to:

- Analyze monthly catalog additions.
- Explore historical trends from 2015 to 2020.
- Compare multiple forecasting algorithms.
- Evaluate model performance using rolling-origin validation.
- Generate monthly projections for 2021–2025.
- Visualize results through an interactive analytics dashboard.

> **Note:** Forecasts are educational, model-generated estimates. They are not official Netflix predictions or verified future catalog counts.

---

## ✨ Key Features

### 📊 Historical Trend Analysis
- Monthly Netflix catalog addition analysis.
- Yearly trend visualization.
- Movie and TV Show distribution.
- Interactive charts and date filters.

### 🤖 Machine Learning & Forecasting
Four forecasting approaches are compared:

1. Naive Baseline
2. Seasonal Naive
3. Linear Regression with trend and seasonal features
4. Holt-Winters Exponential Smoothing

### 🧪 Model Evaluation
- Rolling-origin validation.
- Mean Absolute Error (MAE).
- Root Mean Squared Error (RMSE).
- Mean Absolute Percentage Error (MAPE).
- Comparison of model performance across test years.

### 🔮 Future Forecasting
- Monthly projections from January 2021 through December 2025.
- Forecast horizon selection.
- Interactive forecast visualization.
- Downloadable prediction results.

### 🎨 Interactive Analytics Dashboard
Built with Streamlit and Plotly, featuring:

- Professional dark-themed interface.
- Netflix-inspired red accents with teal, purple, and blue highlights.
- KPI cards and interactive charts.
- Historical Trends page.
- Forecast Explorer page.
- Model Evaluation page.
- Dataset Explorer page.
- CSV export functionality.

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Pandas | Data processing and manipulation |
| NumPy | Numerical calculations |
| Scikit-learn | Linear Regression and evaluation metrics |
| Statsmodels | Holt-Winters time-series forecasting |
| Joblib | Saving and loading model artifacts |
| Plotly | Interactive visualizations |
| Streamlit | Web-based analytics dashboard |
| Matplotlib | Forecast visualization |
| Git & GitHub | Version control and project hosting |

---

## 📂 Project Structure

```text
Netflix_Trend_Forecasting/
│
├── app.py
├── main.py
├── requirements.txt
├── README.md
├── .gitignore
├── Dataset (1).csv
│
├── data/
│   └── netflix_prepared.csv
│
├── models/
│   └── best_model.joblib
│
├── outputs/
│   ├── plots/
│   │   └── forecast.png
│   ├── yearly_trends.csv
│   ├── model_evaluation.csv
│   ├── rolling_validation_results.csv
│   ├── future_predictions.csv
│   ├── monthly_counts.csv
│   └── year_coverage.csv
│
└── src/
    ├── data_preprocessing.py
    ├── data_quality.py
    ├── trend_analysis.py
    ├── forecasting_models.py
    ├── predict_future.py
    └── evaluate_model.py
```

*Some generated files may only exist after running the relevant scripts.*

---

## 📈 Dataset

The project uses a Netflix titles dataset containing movie and TV show metadata, including the date each title was added to the catalog.

### Study Period

- Historical analysis and forecasting experiment: **2015–2020**
- Monthly observations: **72**
- Forecast horizon: **60 months**
- Forecast period: **January 2021–December 2025**

The original dataset includes records beyond the selected training period. The forecasting experiment intentionally uses 2015–2020 for consistent monthly model evaluation.

### Data Preprocessing

The preprocessing workflow includes:

- Parsing the `date_added` column into datetime format.
- Removing records with invalid or missing dates.
- Creating date-based features such as year, month, quarter, and day.
- Aggregating title additions into monthly counts.
- Preparing a dataset for historical analysis and forecasting.

Missing months in the monthly time series are represented as zero counts. This is a modeling assumption and may affect results if the source dataset has incomplete coverage.

---

## 🧠 Forecasting Methodology

### 1. Naive Baseline

Uses the most recent training observation as the prediction for each future month.

### 2. Seasonal Naive

Uses the count from the corresponding month in the previous year. This provides a seasonal benchmark for monthly data.

### 3. Linear Regression

Uses a time index and sine/cosine features to represent trend and monthly seasonality.

### 4. Holt-Winters Exponential Smoothing

Models an additive trend and additive yearly seasonality with a 12-month seasonal period.

### Validation Strategy

Rolling-origin validation evaluates models using historical test years:

- 2018
- 2019
- 2020

For each test year, the model is evaluated using the preceding historical observations. Average validation MAE is used to select the saved forecasting approach.

---

## 📊 Model Evaluation Results

The following results were obtained during the project's rolling-origin validation.

| Model | Average MAE | Average RMSE | Average MAPE |
|---|---:|---:|---:|
| Seasonal Naive | 36.94 | 46.33 | 23.30% |
| Holt-Winters | 39.81 | 44.71 | 27.95% |
| Naive Baseline | 42.92 | 47.96 | 29.34% |
| Linear Regression | 147.46 | 150.22 | 96.32% |

### Best Model

**Seasonal Naive** achieved the lowest average MAE of approximately **36.94** across the three test years.

Holt-Winters achieved a lower average RMSE of approximately **44.71**.

MAE and RMSE are measured in monthly title counts. MAPE is calculated only for observations with non-zero actual values.

These results describe performance on this particular historical dataset and validation setup; they do not guarantee future forecasting accuracy.

---

## 🔮 Forecast Output

The forecasting script generates:

`outputs/future_predictions.csv`

The file contains:

| Column | Description |
|---|---|
| `month` | Forecast month |
| `predicted_titles_added` | Predicted monthly catalog additions |

The saved model artifact is located at:

`models/best_model.joblib`

A forecast chart is generated at:

`outputs/plots/forecast.png`

The current seasonal-naive model repeats the observed monthly pattern from the final training year. As a result, the forecast is seasonal rather than an independent prediction of future Netflix business decisions.

---

## 🚀 Installation and Setup

### Prerequisites

- Python 3.10 or later recommended.
- Git.
- Visual Studio Code or another Python editor.

### Step 1: Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/Netflix_Trend_Forecasting.git
cd Netflix_Trend_Forecasting
```

Replace `YOUR_USERNAME` with your GitHub username.

### Step 2: Create a Virtual Environment

Windows PowerShell:

```powershell
python -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, you can run the Python commands using the environment's interpreter directly.

### Step 3: Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Prepare the Dataset

Ensure the original dataset is located at:

```text
Dataset (1).csv
```

Run:

```bash
python src/data_preprocessing.py
```

### Step 5: Train and Evaluate Forecasting Models

```bash
python src/forecasting_models.py
```

This creates or updates the model evaluation results and saved model artifact.

### Step 6: Generate Future Predictions

```bash
python src/predict_future.py
```

This generates the monthly forecast CSV and forecast chart.

### Step 7: Launch the Dashboard

```bash
streamlit run app.py
```

Open the local URL printed in the terminal, usually:

```text
http://localhost:8501
```

---

## 📁 Generated Files

| File | Purpose |
|---|---|
| `data/netflix_prepared.csv` | Prepared dataset |
| `models/best_model.joblib` | Saved forecasting model artifact |
| `outputs/model_evaluation.csv` | Average model evaluation results |
| `outputs/rolling_validation_results.csv` | Validation scores by test year |
| `outputs/future_predictions.csv` | Monthly forecast predictions |
| `outputs/plots/forecast.png` | Historical and forecast chart |

---

## 🌐 Deployment

The Streamlit dashboard can be deployed using Streamlit Community Cloud or another hosting platform that supports Streamlit applications.

Before deployment:

1. Upload your project to GitHub.
2. Ensure `app.py` and `requirements.txt` are in the repository root.
3. Verify that all required datasets and model files are available to the deployment.
4. Configure the deployment to run `app.py`.
5. Test all dashboard pages after deployment.

If the dataset is large, consider using Git LFS or an appropriate data-hosting approach instead of committing large files directly.

---

## 🔧 Future Improvements

Potential enhancements include:

- Comparing additional forecasting algorithms.
- Using more extensive rolling-origin validation.
- Evaluating alternative seasonal and trend assumptions.
- Adding confidence or prediction intervals where supported by the model.
- Improving data completeness checks.
- Separating actual observations from forecasts in visualizations.
- Adding filters by content type, country, and genre.
- Integrating a refreshed dataset for updated analysis.
- Adding automated tests and a reproducible training pipeline.

---

## ⚠️ Limitations

- The dataset is a static snapshot rather than a live Netflix catalog feed.
- The experiment uses only the 2015–2020 period.
- Missing monthly observations are treated as zero counts.
- Forecasts are based on historical patterns and cannot anticipate unexpected changes in licensing, catalog strategy, or market conditions.
- The forecast period begins in 2021 because the model was trained through December 2020. These projections should not be confused with observed actual counts for subsequent years.
- Validation results depend on the selected years, dataset coverage, and forecasting assumptions.

---

## 👨‍💻 Author

**Sujoy Rana**

B.Tech — Computer Science Engineering, Artificial Intelligence and Machine Learning

Areas of interest:
- Data Science
- Machine Learning
- Time-Series Forecasting
- Data Analytics
- Interactive Data Visualization

---

## 📜 Disclaimer

This project is developed for educational and portfolio purposes. Netflix is a trademark of its respective owner. This project is not affiliated with or endorsed by Netflix.

The analysis and forecasts are experimental and should not be interpreted as official company information or business guidance.

---

⭐ If you find this project useful, consider starring the repository on GitHub.
