
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "netflix_prepared.csv"
MODEL_PATH = BASE_DIR / "models" / "best_model.joblib"
OUTPUT_DIR = BASE_DIR / "outputs"

EVALUATION_PATH = OUTPUT_DIR / "model_evaluation.csv"
ROLLING_PATH = OUTPUT_DIR / "rolling_validation_results.csv"
FORECAST_PATH = OUTPUT_DIR / "future_predictions.csv"

st.set_page_config(
    page_title="Netflix Analytics Studio",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# 2. PROFESSIONAL DARK THEME
# ============================================================

st.markdown("""
<style>
:root {
    --background: #0b0d17;
    --panel: #151a2b;
    --border: #303750;
    --text: #f4f6ff;
    --muted: #a5aec5;
    --red: #e50914;
    --teal: #38d9a9;
    --purple: #a78bfa;
    --blue: #46b8ff;
    --orange: #ffb454;
}

.stApp {
    background:
        radial-gradient(ellipse at top left, #20182e 0%, transparent 38%),
        linear-gradient(135deg, #0b0d17, #111629 55%, #0c1020);
    color: var(--text);
}

.block-container {
    max-width: 1380px;
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}

h1, h2, h3 {
    color: #f4f6ff !important;
    font-weight: 800 !important;
}

p, label, .stMarkdown {
    color: #d0d6e7;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #101321, #171c2e);
    border-right: 1px solid var(--border);
}

[data-testid="stSidebar"] * {
    color: #edf0fa;
}

[data-testid="stMetric"] {
    background: linear-gradient(145deg, #1c2338, #151a2b);
    border: 1px solid var(--border);
    border-radius: 15px;
    padding: 18px;
    box-shadow: 0 7px 22px rgba(0,0,0,.18);
    transition: transform .2s ease, border-color .2s ease;
}

[data-testid="stMetric"]:hover {
    transform: translateY(-3px);
    border-color: var(--red);
}

[data-testid="stMetricLabel"] {
    color: var(--muted) !important;
}

[data-testid="stMetricValue"] {
    color: white !important;
    font-weight: 800;
}

.hero {
    padding: 27px 30px;
    border: 1px solid #35344d;
    border-radius: 18px;
    background:
        radial-gradient(ellipse at right, #47202e 0%, transparent 52%),
        linear-gradient(120deg, #1b2033, #141728);
    margin-bottom: 22px;
}

.hero-eyebrow {
    color: #ff737b;
    font-size: 12px;
    letter-spacing: 2px;
    font-weight: 800;
    text-transform: uppercase;
}

.hero-title {
    color: #ffffff;
    font-size: clamp(27px, 4vw, 42px);
    font-weight: 850;
    line-height: 1.2;
    margin-top: 10px;
}

.hero-description {
    color: #bac3d8;
    font-size: 15px;
    line-height: 1.7;
    margin-top: 12px;
}

.section-heading {
    color: #f4f6ff;
    font-size: 21px;
    font-weight: 750;
    margin: 22px 0 12px;
}

.panel {
    background: linear-gradient(145deg, #1b2235, #14192a);
    border: 1px solid var(--border);
    border-radius: 15px;
    padding: 17px 19px;
    margin-bottom: 12px;
}

.panel-label {
    color: #a5aec5;
    font-size: 12px;
    letter-spacing: 1px;
    text-transform: uppercase;
}

.panel-value {
    color: white;
    font-size: 25px;
    font-weight: 800;
    margin-top: 7px;
    overflow-wrap: anywhere;
}

.panel-note {
    color: #9ba7c1;
    font-size: 12px;
    margin-top: 5px;
}

.red-text { color: #ff646d; }
.teal-text { color: #38d9a9; }
.purple-text { color: #b6a2ff; }
.blue-text { color: #65c7ff; }
.orange-text { color: #ffbd70; }

.stButton > button,
.stDownloadButton > button {
    background: linear-gradient(90deg, #e50914, #b20710);
    color: white !important;
    border: 1px solid #e50914;
    border-radius: 10px;
    min-height: 42px;
    font-weight: 700;
    transition: all .2s ease;
}

.stButton > button:hover,
.stDownloadButton > button:hover {
    background: #ff2533;
    border-color: #ff2533;
    box-shadow: 0 5px 18px rgba(229,9,20,.25);
}

.stTextInput input,
.stNumberInput input {
    background: #171d2e !important;
    color: white !important;
    border: 1px solid #3a435c !important;
    border-radius: 9px !important;
}

[data-baseweb="select"] > div {
    background: #171d2e;
    border-color: #3a435c;
}

[data-testid="stAlert"] {
    background: #19243a;
    border: 1px solid #334665;
    border-radius: 12px;
}

[data-testid="stCaptionContainer"] {
    color: #9ba5bd !important;
}

hr {
    border-color: #303750;
}

@media (max-width: 768px) {
    .block-container {
        padding: 1rem .8rem;
    }
    .hero {
        padding: 20px;
    }
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# 3. COLORS AND CHART HELPERS
# ============================================================

COLORS = {
    "red": "#E50914",
    "teal": "#38D9A9",
    "purple": "#A78BFA",
    "blue": "#46B8FF",
    "orange": "#FFB454",
    "pink": "#FF70A6",
}

PALETTE = list(COLORS.values())


def style_chart(fig, height=390):
    fig.update_layout(
        template="plotly_dark",
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(
            family="Arial, sans-serif",
            color="#DCE3F5",
            size=12,
        ),
        title=dict(
            font=dict(size=17, color="#F4F6FF"),
            x=0.02,
        ),
        margin=dict(l=15, r=15, t=65, b=15),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color="#DCE3F5", size=11),
        ),
        hoverlabel=dict(
            bgcolor="#202840",
            font_color="#FFFFFF",
        ),
    )
    fig.update_xaxes(
        showgrid=False,
        zeroline=False,
        linecolor="#303750",
    )
    fig.update_yaxes(
        gridcolor="rgba(140,155,185,0.13)",
        zeroline=False,
    )
    return fig


def hero(title, description, eyebrow="NETFLIX ANALYTICS STUDIO"):
    st.markdown(f"""
    <div class="hero">
        <div class="hero-eyebrow">{eyebrow}</div>
        <div class="hero-title">{title}</div>
        <div class="hero-description">{description}</div>
    </div>
    """, unsafe_allow_html=True)


def panel_metric(label, value, note="", color="teal-text"):
    st.markdown(f"""
    <div class="panel">
        <div class="panel-label">{label}</div>
        <div class="panel-value {color}">{value}</div>
        <div class="panel-note">{note}</div>
    </div>
    """, unsafe_allow_html=True)


def dark_table_style(dataframe):
    return (
        dataframe.style
        .set_properties(**{
            "background-color": "#151a2b",
            "color": "#e8edfa",
            "border-color": "#303750",
        })
        .set_table_styles([
            {
                "selector": "th",
                "props": [
                    ("background-color", "#222a40"),
                    ("color", "#a9b9d8"),
                    ("font-weight", "bold"),
                    ("border-color", "#303750"),
                    ("padding", "10px"),
                ],
            },
            {
                "selector": "td",
                "props": [
                    ("background-color", "#151a2b"),
                    ("color", "#e8edfa"),
                    ("border-color", "#303750"),
                    ("padding", "9px"),
                ],
            },
            {
                "selector": "table",
                "props": [
                    ("border-collapse", "collapse"),
                    ("width", "100%"),
                ],
            },
        ])
    )


# ============================================================
# 4. LOAD DATA
# ============================================================

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH, parse_dates=["date_added"])
    df = df.dropna(subset=["date_added"])
    df = df[df["date_added"].dt.year.between(2015, 2020)].copy()

    monthly = (
        df.set_index("date_added")
        .resample("MS")
        .size()
        .rename("titles_added")
        .reset_index()
        .rename(columns={"date_added": "month"})
    )

    return df, monthly


@st.cache_data
def load_evaluation():
    if EVALUATION_PATH.exists():
        return pd.read_csv(EVALUATION_PATH)
    return pd.DataFrame()


@st.cache_data
def load_rolling_results():
    if ROLLING_PATH.exists():
        return pd.read_csv(ROLLING_PATH)
    return pd.DataFrame()


@st.cache_data
def load_forecast():
    if FORECAST_PATH.exists():
        return pd.read_csv(FORECAST_PATH, parse_dates=["month"])
    return pd.DataFrame()


if not DATA_PATH.exists():
    st.error("Prepared dataset not found. Run your preprocessing script first.")
    st.stop()

if not MODEL_PATH.exists():
    st.error("Trained model not found. Run python src/forecasting_models.py.")
    st.stop()

try:
    df, monthly = load_data()
    evaluation = load_evaluation()
    rolling = load_rolling_results()
    forecast = load_forecast()
    model_artifact = joblib.load(MODEL_PATH)
except Exception as exc:
    st.error(f"Could not load project files: {exc}")
    st.stop()

best_model_name = model_artifact.get("best_model_name", "Unknown")


# ============================================================
# 5. SIDEBAR NAVIGATION
# ============================================================

with st.sidebar:
    st.markdown("""
    <div style="padding:8px 0 18px;">
        <div style="font-size:38px;">🎬</div>
        <div style="font-size:23px;font-weight:850;color:#FFFFFF;">
            NETFLIX
        </div>
        <div style="font-size:13px;color:#A5AEC5;">
            Analytics Studio
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    page = st.radio(
        "WORKSPACE",
        [
            "Overview",
            "Historical Trends",
            "Forecast Explorer",
            "Model Evaluation",
            "Dataset Explorer",
        ],
    )

    st.divider()
    st.markdown("### 🧠 Active Model")

    st.markdown(f"""
    <div class="panel">
        <div class="panel-label">Selected algorithm</div>
        <div style="font-size:18px;font-weight:750;color:#FFFFFF;
                    margin-top:8px;">{best_model_name}</div>
        <div class="panel-note">Selected using average validation MAE</div>
    </div>
    """, unsafe_allow_html=True)

    st.caption("Historical study: 2015–2020")
    st.caption("Forecast frequency: Monthly")

    if st.button("↻ Refresh dashboard", use_container_width=True):
        st.cache_data.clear()
        st.rerun()


# ============================================================
# 6. OVERVIEW
# ============================================================

if page == "Overview":
    hero(
        "Netflix Trend Forecasting",
        "Explore catalog additions, historical patterns, forecasting "
        "models, and future projections in one interactive workspace.",
    )

    total_titles = len(df)
    movie_count = int((df["type"] == "Movie").sum()) if "type" in df.columns else 0
    tv_count = int((df["type"] == "TV Show").sum()) if "type" in df.columns else 0
    average_monthly = monthly["titles_added"].mean()

    k1, k2, k3, k4 = st.columns(4)

    with k1:
        panel_metric(
            "TITLES ANALYZED", f"{total_titles:,}",
            "Records in study period", "red-text"
        )
    with k2:
        panel_metric(
            "MOVIES", f"{movie_count:,}",
            "Movie records", "orange-text"
        )
    with k3:
        panel_metric(
            "TV SHOWS", f"{tv_count:,}",
            "TV Show records", "purple-text"
        )
    with k4:
        panel_metric(
            "MONTHLY AVERAGE", f"{average_monthly:,.1f}",
            "Historical additions per month", "teal-text"
        )

    st.markdown(
        '<div class="section-heading">📈 Historical Performance</div>',
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.6, 1])

    with left:
        fig = px.area(
            monthly,
            x="month",
            y="titles_added",
            title="Monthly Catalog Additions",
        )
        fig.update_traces(
            line=dict(color=COLORS["teal"], width=2.5),
            fillcolor="rgba(56,217,169,0.13)",
        )
        fig.update_layout(
            xaxis_title="Month",
            yaxis_title="Titles added",
            hovermode="x unified",
        )
        st.plotly_chart(style_chart(fig), use_container_width=True)

    # -------- FIXED PIE / DONUT CHART --------
    with right:
        type_counts = pd.DataFrame({
            "Content Type": ["Movies", "TV Shows"],
            "Count": [movie_count, tv_count],
        })

        fig = px.pie(
            type_counts,
            names="Content Type",
            values="Count",
            hole=0.62,
            color="Content Type",
            color_discrete_map={
                "Movies": COLORS["red"],
                "TV Shows": COLORS["purple"],
            },
            title="Content Type Distribution",
        )

        fig.update_traces(
            textposition="inside",
            textinfo="label+percent",
            textfont=dict(
                color="#FFFFFF",
                size=14,
                family="Arial",
            ),
            insidetextorientation="horizontal",
            marker=dict(
                line=dict(color="#151a2b", width=3),
            ),
            hovertemplate=(
                "<b>%{label}</b><br>"
                "Titles: %{value:,}<br>"
                "Share: %{percent}<extra></extra>"
            ),
            sort=False,
        )

        fig.update_layout(
            showlegend=False,
            uniformtext_minsize=11,
            uniformtext_mode="hide",
            annotations=[
                dict(
                    text="CONTENT<br><b>MIX</b>",
                    x=0.5,
                    y=0.5,
                    showarrow=False,
                    font=dict(size=15, color="#FFFFFF"),
                )
            ],
        )

        st.plotly_chart(
            style_chart(fig, 390),
            use_container_width=True,
            config={"displayModeBar": False},
        )

    st.markdown(
        '<div class="section-heading">🔮 Forecast Preview</div>',
        unsafe_allow_html=True,
    )

    if not forecast.empty:
        preview = forecast.head(24).copy()

        fig = px.line(
            preview,
            x="month",
            y="predicted_titles_added",
            markers=True,
            title="First 24 Forecast Months",
        )
        fig.update_traces(
            line=dict(color=COLORS["red"], width=3),
            marker=dict(size=6, color=COLORS["orange"]),
        )
        fig.update_layout(
            xaxis_title="Month",
            yaxis_title="Predicted additions",
            hovermode="x unified",
        )

        st.plotly_chart(
            style_chart(fig, 350),
            use_container_width=True,
        )
    else:
        st.warning("Forecast file missing. Run python src/predict_future.py.")

    st.caption(
        "Projections are estimates generated by the saved model, "
        "not verified future Netflix catalog counts."
    )


# ============================================================
# 7. HISTORICAL TRENDS
# ============================================================

elif page == "Historical Trends":
    hero(
        "Historical Trends",
        "Investigate monthly and yearly catalog additions with interactive "
        "filters and multiple chart styles.",
        "HISTORICAL ANALYSIS",
    )

    c1, c2 = st.columns(2)

    with c1:
        year_range = st.slider(
            "Select years",
            min_value=2015,
            max_value=2020,
            value=(2015, 2020),
        )

    with c2:
        chart_type = st.selectbox(
            "Monthly chart style", ["Area", "Line", "Bar"]
        )

    filtered = monthly[
        monthly["month"].dt.year.between(year_range[0], year_range[1])
    ].copy()

    if chart_type == "Area":
        fig = px.area(
            filtered, x="month", y="titles_added",
            title="Monthly Catalog Additions"
        )
        fig.update_traces(
            line=dict(color=COLORS["teal"], width=2.5),
            fillcolor="rgba(56,217,169,0.14)",
        )
    elif chart_type == "Line":
        fig = px.line(
            filtered, x="month", y="titles_added",
            markers=True, title="Monthly Catalog Additions"
        )
        fig.update_traces(line=dict(color=COLORS["blue"], width=3))
    else:
        fig = px.bar(
            filtered, x="month", y="titles_added",
            title="Monthly Catalog Additions"
        )
        fig.update_traces(marker_color=COLORS["red"])

    fig.update_layout(
        xaxis_title="Month",
        yaxis_title="Titles added",
    )
    st.plotly_chart(style_chart(fig, 430), use_container_width=True)

    yearly = (
        filtered.assign(year=filtered["month"].dt.year)
        .groupby("year", as_index=False)["titles_added"]
        .sum()
    )

    left, right = st.columns(2)

    with left:
        fig = px.bar(
            yearly,
            x="year",
            y="titles_added",
            color="titles_added",
            color_continuous_scale=["#29233F", "#9B8CFF", "#E50914"],
            title="Yearly Additions",
        )
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(style_chart(fig, 350), use_container_width=True)

    with right:
        box_data = filtered.assign(
            month_name=filtered["month"].dt.month_name()
        )
        fig = px.box(
            box_data,
            x="month_name",
            y="titles_added",
            title="Distribution of Monthly Counts",
            color_discrete_sequence=[COLORS["pink"]],
        )
        st.plotly_chart(style_chart(fig, 350), use_container_width=True)

    st.markdown("### 📋 Monthly Data")
    st.dataframe(
        dark_table_style(filtered),
        use_container_width=True,
        hide_index=True,
        height=360,
    )

    st.download_button(
        "⬇️ Download historical data",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name="netflix_historical_trends.csv",
        mime="text/csv",
    )


# ============================================================
# 8. FORECAST EXPLORER
# ============================================================

elif page == "Forecast Explorer":
    hero(
        "Forecast Explorer",
        "Inspect monthly projections, change the display horizon, and "
        "download forecast results for further analysis.",
        "PREDICTIVE ANALYTICS",
    )

    if forecast.empty:
        st.warning("Forecast data not found. Run python src/predict_future.py.")
        st.stop()

    forecast = forecast.copy()
    forecast["month"] = pd.to_datetime(forecast["month"])
    available_months = len(forecast)

    horizon = st.slider(
        "Forecast horizon in months",
        min_value=1,
        max_value=available_months,
        value=min(12, available_months),
    )

    selected = forecast.head(horizon).copy()

    k1, k2, k3 = st.columns(3)

    with k1:
        panel_metric(
            "FORECAST HORIZON", f"{horizon} months",
            "Selected projection window", "purple-text"
        )
    with k2:
        panel_metric(
            "TOTAL PREDICTED",
            f"{selected['predicted_titles_added'].sum():,}",
            "Sum across selected months", "teal-text"
        )
    with k3:
        panel_metric(
            "MONTHLY AVERAGE",
            f"{selected['predicted_titles_added'].mean():,.1f}",
            "Average predicted additions", "orange-text"
        )

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=selected["month"],
        y=selected["predicted_titles_added"],
        mode="lines+markers",
        name="Forecast",
        line=dict(color=COLORS["red"], width=3),
        marker=dict(color=COLORS["orange"], size=7),
        fill="tozeroy",
        fillcolor="rgba(229,9,20,0.10)",
        hovertemplate=(
            "<b>%{x|%b %Y}</b><br>"
            "Predicted additions: %{y}<extra></extra>"
        ),
    ))
    fig.update_layout(
        title="Monthly Forecast Projection",
        xaxis_title="Month",
        yaxis_title="Predicted titles",
        hovermode="x unified",
    )
    st.plotly_chart(style_chart(fig, 420), use_container_width=True)

    st.markdown("### 🗓️ Forecast Table")

    table_search = st.text_input(
        "Filter by year",
        placeholder="For example, 2023",
    )

    table_data = selected.copy()

    if table_search.strip():
        year_strings = table_data["month"].dt.year.astype(str)
        table_data = table_data[
            year_strings.str.contains(
                table_search.strip(), regex=False
            )
        ]

    display_table = table_data.copy()
    display_table["month"] = display_table["month"].dt.strftime("%b %Y")
    display_table = display_table.rename(columns={
        "month": "Forecast Month",
        "predicted_titles_added": "Predicted Additions",
    })

    st.dataframe(
        dark_table_style(display_table),
        use_container_width=True,
        hide_index=True,
        height=360,
    )

    st.download_button(
        "⬇️ Download selected forecast CSV",
        data=table_data.to_csv(index=False).encode("utf-8"),
        file_name="netflix_forecast.csv",
        mime="text/csv",
    )

    st.info(
        f"Saved model: {best_model_name}. The horizon slider changes the "
        "displayed predictions; it does not retrain the forecasting model."
    )

    st.caption(
        "These are model-generated estimates based on historical patterns, "
        "not official or verified Netflix release counts."
    )


# ============================================================
# 9. MODEL EVALUATION
# ============================================================

elif page == "Model Evaluation":
    hero(
        "Model Evaluation Lab",
        "Compare forecasting algorithms using rolling-origin validation "
        "and investigate prediction errors across test years.",
        "MODEL PERFORMANCE",
    )

    if evaluation.empty:
        st.warning(
            "Evaluation file missing. Run python src/forecasting_models.py."
        )
        st.stop()

    metric = st.selectbox(
        "Choose evaluation metric", ["MAE", "RMSE", "MAPE_percent"]
    )

    best_mae_model = evaluation.sort_values("MAE").iloc[0]
    metric_winner = evaluation.sort_values(metric).iloc[0]

    a, b, c = st.columns(3)

    with a:
        panel_metric(
            "BEST MODEL BY MAE",
            str(best_mae_model["model"]),
            "Lowest average MAE",
            "teal-text",
        )
    with b:
        panel_metric(
            f"LOWEST {metric}",
            f"{metric_winner[metric]:.2f}",
            "Across average validation scores",
            "purple-text",
        )
    with c:
        panel_metric(
            "MODELS COMPARED",
            str(len(evaluation)),
            "Forecasting algorithms",
            "orange-text",
        )

    fig = px.bar(
        evaluation.sort_values(metric),
        x="model",
        y=metric,
        color="model",
        text_auto=".2f",
        title=f"Model Comparison — {metric}",
        color_discrete_sequence=PALETTE,
    )
    fig.update_layout(
        xaxis_title="Model",
        yaxis_title=metric,
        showlegend=False,
    )
    st.plotly_chart(style_chart(fig, 410), use_container_width=True)

    st.markdown("### 📊 Average Validation Scores")
    st.dataframe(
        dark_table_style(evaluation),
        use_container_width=True,
        hide_index=True,
        height=250,
    )

    if not rolling.empty:
        st.markdown("### 🧪 Performance Across Test Years")

        fig = px.line(
            rolling,
            x="test_year",
            y="MAE",
            color="model",
            markers=True,
            title="MAE Across Test Years",
            color_discrete_sequence=PALETTE,
        )
        fig.update_layout(
            xaxis_title="Test year",
            yaxis_title="MAE",
            hovermode="x unified",
        )
        st.plotly_chart(style_chart(fig, 410), use_container_width=True)

        st.dataframe(
            dark_table_style(rolling.sort_values(["test_year", "model"])),
            use_container_width=True,
            hide_index=True,
            height=350,
        )

        st.download_button(
            "⬇️ Download rolling validation results",
            data=rolling.to_csv(index=False).encode("utf-8"),
            file_name="rolling_validation_results.csv",
            mime="text/csv",
        )

    st.caption(
        "Lower values are better. MAE and RMSE are measured in titles. "
        "MAPE is calculated only for months with non-zero actual counts."
    )


# ============================================================
# 10. DATASET EXPLORER
# ============================================================

elif page == "Dataset Explorer":
    hero(
        "Dataset Explorer",
        "Search, filter, inspect, and export the records used in the "
        "historical forecasting experiment.",
        "DATA MANAGEMENT",
    )

    filtered_df = df.copy()

    c1, c2 = st.columns([1.4, 1])

    with c1:
        search_text = st.text_input(
            "🔎 Search titles",
            placeholder="Enter a movie or TV show title...",
        )

    with c2:
        if "type" in filtered_df.columns:
            types = sorted(filtered_df["type"].dropna().unique().tolist())
            selected_types = st.multiselect(
                "Content type",
                options=types,
                default=types,
            )
        else:
            selected_types = []

    if search_text.strip() and "title" in filtered_df.columns:
        filtered_df = filtered_df[
            filtered_df["title"].str.contains(
                search_text.strip(),
                case=False,
                na=False,
                regex=False,
            )
        ]

    if "type" in filtered_df.columns:
        if selected_types:
            filtered_df = filtered_df[
                filtered_df["type"].isin(selected_types)
            ]
        else:
            filtered_df = filtered_df.iloc[0:0]

    k1, k2, k3 = st.columns(3)

    with k1:
        panel_metric(
            "FILTERED RECORDS", f"{len(filtered_df):,}",
            "Matching selected filters", "teal-text"
        )
    with k2:
        panel_metric(
            "AVAILABLE COLUMNS", f"{len(filtered_df.columns)}",
            "Dataset fields", "purple-text"
        )
    with k3:
        panel_metric(
            "DATE COVERAGE", "2015–2020",
            "Historical study period", "orange-text"
        )

    st.markdown("### 📋 Records")
    st.dataframe(
        dark_table_style(filtered_df),
        use_container_width=True,
        hide_index=True,
        height=430,
    )

    st.download_button(
        "⬇️ Download filtered dataset CSV",
        data=filtered_df.to_csv(index=False).encode("utf-8"),
        file_name="netflix_filtered_dataset.csv",
        mime="text/csv",
    )

    st.markdown("### 🧾 Dataset Information")

    left, right = st.columns(2)

    with left:
        st.markdown("**Column names**")
        st.write(", ".join(map(str, df.columns.tolist())))

    with right:
        st.markdown("**Missing values by column**")
        missing = (
            df.isna().sum()
            .rename("missing_values")
            .reset_index()
            .rename(columns={"index": "column"})
        )
        st.dataframe(
            dark_table_style(missing),
            use_container_width=True,
            hide_index=True,
            height=300,
        )


# ============================================================
# 11. FOOTER
# ============================================================

st.divider()

left, right = st.columns([2, 1])

with left:
    st.caption(
        "🎬 Netflix Analytics Studio | Machine Learning & Time-Series "
        "Forecasting Project"
    )

with right:
    st.caption("Python · Streamlit · Plotly")

st.caption(
    "Educational project. Historical catalog additions and future "
    "projections should not be interpreted as official Netflix data."
)
