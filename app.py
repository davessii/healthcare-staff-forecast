"""Streamlit app for healthcare staff forecasting.

Run from the project root:
    python -m streamlit run app.py
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import sklearn
import streamlit as st

from model_training import PROJECT_VERSION, train_all
from src.features import (
    BASE_INDICATORS,
    METRIC_COLUMNS,
    RUSSIAN_COLUMNS,
    TARGET_NAMES,
    add_ml_features,
    clean_feature_rows,
    rename_for_display,
)

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "processed" / "healthcare_dataset_model_ready.csv"
MODELS_DIR = ROOT / "models"
REPORTS_PATH = ROOT / "reports" / "model_metrics.csv"
METADATA_PATH = MODELS_DIR / "model_metadata.json"

st.set_page_config(page_title="Прогноз кадров здравоохранения", layout="wide")

st.markdown(
    """
    <style>
    a[href^="#"] {display: none !important;}
    .main .block-container {padding-top: 2rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    return df.sort_values(["region", "year"]).reset_index(drop=True)


def model_artifacts_are_current() -> bool:
    required_files = [
        MODELS_DIR / "doctors_model.pkl",
        MODELS_DIR / "nurses_model.pkl",
        METADATA_PATH,
        REPORTS_PATH,
    ]
    if not all(path.exists() for path in required_files):
        return False

    try:
        with open(METADATA_PATH, "r", encoding="utf-8") as file:
            metadata = json.load(file)
    except Exception:
        return False

    return (
        metadata.get("project_version") == PROJECT_VERSION
        and metadata.get("sklearn_version") == sklearn.__version__
    )


@st.cache_resource
def load_models():
    if not model_artifacts_are_current():
        train_all()

    doctors_model = joblib.load(MODELS_DIR / "doctors_model.pkl")
    nurses_model = joblib.load(MODELS_DIR / "nurses_model.pkl")
    with open(METADATA_PATH, "r", encoding="utf-8") as file:
        metadata = json.load(file)
    return doctors_model, nurses_model, metadata


def show_dataframe(df: pd.DataFrame, **kwargs) -> None:
    """Use the modern Streamlit width API, with a fallback for older versions."""
    try:
        st.dataframe(df, width="stretch", **kwargs)
    except TypeError:
        st.dataframe(df, use_container_width=True, **kwargs)


def show_plot(fig) -> None:
    try:
        st.plotly_chart(fig, width="stretch")
    except TypeError:
        st.plotly_chart(fig, use_container_width=True)


def line_chart_long(
    df: pd.DataFrame,
    value_columns: list[str],
    title: str,
    y_title: str,
    labels: dict[str, str] | None = None,
) -> None:
    chart_df = df[["year", *value_columns]].copy()
    chart_df = chart_df.rename(columns=labels or {})
    renamed_value_columns = [labels.get(col, col) if labels else col for col in value_columns]
    chart_df["Год"] = chart_df["year"].astype(str)
    long_df = chart_df.melt(
        id_vars="Год",
        value_vars=renamed_value_columns,
        var_name="Показатель",
        value_name=y_title,
    )
    fig = px.line(
        long_df,
        x="Год",
        y=y_title,
        color="Показатель",
        markers=True,
        title=title,
    )
    fig.update_layout(hovermode="x unified", legend_title_text="")
    fig.update_xaxes(type="category")
    show_plot(fig)


def forecast_chart(history: pd.DataFrame, forecast: pd.DataFrame, y_columns: list[str], title: str) -> None:
    fact = history[["year", *y_columns]].copy()
    fact["Тип"] = "факт"
    pred = forecast[["year", *y_columns]].copy()
    pred["Тип"] = "прогноз"
    chart_df = pd.concat([fact, pred], ignore_index=True)
    chart_df["Год"] = chart_df["year"].astype(str)
    chart_df = chart_df.rename(columns={col: RUSSIAN_COLUMNS.get(col, col) for col in y_columns})
    renamed_columns = [RUSSIAN_COLUMNS.get(col, col) for col in y_columns]
    long_df = chart_df.melt(
        id_vars=["Год", "Тип"],
        value_vars=renamed_columns,
        var_name="Показатель",
        value_name="Значение",
    )
    fig = px.line(
        long_df,
        x="Год",
        y="Значение",
        color="Показатель",
        line_dash="Тип",
        markers=True,
        title=title,
    )
    fig.update_layout(hovermode="x unified", legend_title_text="")
    fig.update_xaxes(type="category")
    show_plot(fig)


def robust_growth(series: pd.Series, window: int = 5, max_abs: float = 0.03) -> float:
    """Return a stable annual growth estimate for recursive forecasting.

    Multi-year recursive ML forecasts can accumulate one-year model errors and
    produce unrealistic jumps. For diploma-grade interpretability, the forecast
    block uses the ML prediction together with a robust regional trend estimated
    from the last observed years.
    """
    values = pd.to_numeric(series, errors="coerce").dropna()
    growth = values.pct_change().replace([np.inf, -np.inf], np.nan).dropna().tail(window)
    if growth.empty:
        return 0.0
    return float(np.clip(growth.median(), -max_abs, max_abs))


def bounded_projection(series: pd.Series, step: int, max_abs_growth: float, min_value: float = 0.0) -> float:
    """Project a non-target indicator by a bounded robust trend."""
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return min_value
    rate = robust_growth(values, window=5, max_abs=max_abs_growth)
    return max(float(values.iloc[-1]) * ((1.0 + rate) ** step), min_value)


def stabilized_staff_forecast(
    raw_prediction: float,
    observed_series: pd.Series,
    step: int,
    model_weight: float = 0.25,
    max_abs_growth: float = 0.03,
    max_deviation_from_trend: float = 0.04,
) -> float:
    """Blend ML output with a stable regional trend and clip implausible jumps.

    The ML model remains the main predictive component in the system, but for
    multi-step forecasts the raw one-year model prediction is post-processed.
    This prevents abrupt growth/decline when the model is recursively fed its
    own future predictions.
    """
    values = pd.to_numeric(observed_series, errors="coerce").dropna()
    if values.empty:
        return max(float(raw_prediction), 0.0)

    base_value = float(values.iloc[-1])
    regional_rate = robust_growth(values, window=5, max_abs=max_abs_growth)
    trend_value = base_value * ((1.0 + regional_rate) ** step)

    blended = model_weight * float(raw_prediction) + (1.0 - model_weight) * trend_value
    lower = trend_value * (1.0 - max_deviation_from_trend)
    upper = trend_value * (1.0 + max_deviation_from_trend)

    return max(float(np.clip(blended, lower, upper)), 0.0)


def build_next_row(
    history: pd.DataFrame,
    observed_history: pd.DataFrame,
    step: int,
    doctors_model,
    nurses_model,
    metadata: dict,
) -> dict:
    history = history.sort_values("year").copy()
    feature_table = add_ml_features(history, add_targets=False)
    feature_table = clean_feature_rows(feature_table)
    current_features = feature_table.tail(1)

    doctors_features = metadata["targets"]["doctors"]["feature_columns"]
    nurses_features = metadata["targets"]["nurses"]["feature_columns"]

    raw_doctors = float(doctors_model.predict(current_features[doctors_features])[0])
    raw_nurses = float(nurses_model.predict(current_features[nurses_features])[0])

    next_doctors = stabilized_staff_forecast(
        raw_doctors,
        observed_history["doctors_count"],
        step=step,
        model_weight=0.25,
        max_abs_growth=0.025,
        max_deviation_from_trend=0.04,
    )
    next_nurses = stabilized_staff_forecast(
        raw_nurses,
        observed_history["nurses_count"],
        step=step,
        model_weight=0.25,
        max_abs_growth=0.03,
        max_deviation_from_trend=0.04,
    )

    next_year = int(history["year"].max()) + 1
    population = bounded_projection(observed_history["population"], step, max_abs_growth=0.015, min_value=1.0)
    visits_count = bounded_projection(observed_history["visits_count"], step, max_abs_growth=0.04)
    hospital_beds = bounded_projection(observed_history["hospital_beds"], step, max_abs_growth=0.04)
    outpatient_capacity = bounded_projection(observed_history["outpatient_capacity"], step, max_abs_growth=0.06)

    row = {
        "region": history["region"].iloc[-1],
        "year": next_year,
        "population": round(population),
        "doctors_count": round(next_doctors),
        "nurses_count": round(next_nurses),
        "visits_count": visits_count,
        "hospital_beds": round(max(hospital_beds, 0.0)),
        "outpatient_capacity": outpatient_capacity,
    }

    row["doctors_per_10000"] = row["doctors_count"] / row["population"] * 10000
    row["nurses_per_10000"] = row["nurses_count"] / row["population"] * 10000
    row["beds_per_10000"] = row["hospital_beds"] / row["population"] * 10000
    row["visits_per_person"] = row["visits_count"] / row["population"]
    return row


def make_forecast(df: pd.DataFrame, region: str, horizon: int, doctors_model, nurses_model, metadata: dict) -> pd.DataFrame:
    history = df[df["region"] == region].sort_values("year").copy()
    observed_history = history.copy()
    future_rows = []
    work = history.copy()

    for step in range(1, horizon + 1):
        next_row = build_next_row(work, observed_history, step, doctors_model, nurses_model, metadata)
        future_rows.append(next_row)
        work = pd.concat([work, pd.DataFrame([next_row])], ignore_index=True)

    return pd.DataFrame(future_rows)


def status_from_deficit(deficit: float, need: float) -> str:
    if need <= 0:
        return "нет данных"
    rel = deficit / need
    if rel > 0.05:
        return "дефицит кадров"
    if rel < -0.05:
        return "избыток кадров"
    return "близко к нормативу"


def status_emoji(status: str) -> str:
    if status == "дефицит кадров":
        return "🔴 дефицит кадров"
    if status == "избыток кадров":
        return "🔵 избыток кадров"
    if status == "близко к нормативу":
        return "🟢 близко к нормативу"
    return status


def build_need_table(forecast_df: pd.DataFrame, doctors_norm: float, nurses_norm: float) -> pd.DataFrame:
    result = forecast_df.copy()
    result["doctors_need"] = result["population"] * doctors_norm / 10000
    result["nurses_need"] = result["population"] * nurses_norm / 10000
    result["doctors_deficit"] = result["doctors_need"] - result["doctors_count"]
    result["nurses_deficit"] = result["nurses_need"] - result["nurses_count"]
    result["doctors_status"] = result.apply(lambda r: status_from_deficit(r["doctors_deficit"], r["doctors_need"]), axis=1)
    result["nurses_status"] = result.apply(lambda r: status_from_deficit(r["nurses_deficit"], r["nurses_need"]), axis=1)
    result["doctors_status"] = result["doctors_status"].map(status_emoji)
    result["nurses_status"] = result["nurses_status"].map(status_emoji)
    return result


def display_forecast_table(forecast_df: pd.DataFrame) -> None:
    columns = [
        "year",
        "population",
        "doctors_count",
        "nurses_count",
        "doctors_per_10000",
        "nurses_per_10000",
        "hospital_beds",
        "beds_per_10000",
        "visits_per_person",
        "outpatient_capacity",
    ]
    table = forecast_df[columns].round(2)
    show_dataframe(rename_for_display(table), hide_index=True)


def display_need_table(result: pd.DataFrame) -> None:
    columns = [
        "year",
        "population",
        "doctors_count",
        "doctors_need",
        "doctors_deficit",
        "doctors_status",
        "nurses_count",
        "nurses_need",
        "nurses_deficit",
        "nurses_status",
    ]
    table = result[columns].round(2)
    show_dataframe(rename_for_display(table), hide_index=True)


def main() -> None:
    if not DATA_PATH.exists():
        st.info("Сначала создайте учебные данные: python scripts/make_demo_data.py")
        st.stop()
    df = load_data()
    if df['region'].astype(str).str.startswith('Демо регион ').all():
        st.warning("Демонстрационные искусственные данные. Это не статистика реальных регионов и не рекомендация по кадровым решениям.")
    doctors_model, nurses_model, metadata = load_models()
    regions = sorted(df["region"].unique())
    default_region = regions.index("Ростовская область") if "Ростовская область" in regions else 0

    st.title("Информационная система прогнозирования кадров здравоохранения")
    page = st.sidebar.radio(
        "Раздел",
        ["Главная", "Анализ данных", "Прогнозирование", "Оценка потребности", "Сравнение моделей"],
    )

    if page == "Главная":
        st.subheader("Назначение системы")
        st.write(
            "Система предназначена для анализа статистических данных о кадровом обеспечении "
            "здравоохранения регионов Российской Федерации и прогнозирования численности врачей "
            "и среднего медицинского персонала методами машинного обучения."
        )

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Регионов", df["region"].nunique())
        col2.metric("Период", f"{df['year'].min()}–{df['year'].max()}")
        col3.metric("Наблюдений", len(df))
        col4.metric("Версия", PROJECT_VERSION)

        st.subheader("Основные показатели")
        indicator_names = pd.DataFrame(
            {
                "Показатель в датасете": [RUSSIAN_COLUMNS.get(col, col) for col in BASE_INDICATORS],
                "Техническое имя": BASE_INDICATORS,
            }
        )
        show_dataframe(indicator_names, hide_index=True)

        st.subheader("Лучшие модели")
        best_models = pd.DataFrame(
            [
                {
                    "Целевой показатель": TARGET_NAMES[key],
                    "Выбранная модель": metadata["targets"][key]["best_model"],
                }
                for key in ["doctors", "nurses"]
            ]
        )
        show_dataframe(best_models, hide_index=True)

    elif page == "Анализ данных":
        region = st.selectbox("Регион", regions, index=default_region)
        region_df = df[df["region"] == region].sort_values("year")
        last = region_df.iloc[-1]

        st.subheader(f"Динамика показателей: {region}")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Врачей, послед. год", f"{last['doctors_count']:,.0f}".replace(",", " "))
        col2.metric("Среднего медперсонала", f"{last['nurses_count']:,.0f}".replace(",", " "))
        col3.metric("Врачей на 10 000", f"{last['doctors_per_10000']:.2f}")
        col4.metric("Среднего медперсонала на 10 000", f"{last['nurses_per_10000']:.2f}")

        display_cols = [
            "region",
            "year",
            "population",
            "doctors_count",
            "nurses_count",
            "visits_count",
            "hospital_beds",
            "outpatient_capacity",
            "doctors_per_10000",
            "nurses_per_10000",
            "beds_per_10000",
            "visits_per_person",
        ]
        show_dataframe(rename_for_display(region_df[display_cols].round(3)), hide_index=True)

        line_chart_long(
            region_df,
            ["doctors_count", "nurses_count"],
            f"Численность медицинских кадров: {region}",
            "Численность, человек",
            labels={
                "doctors_count": "Врачи",
                "nurses_count": "Средний медперсонал",
            },
        )
        line_chart_long(
            region_df,
            ["doctors_per_10000", "nurses_per_10000", "beds_per_10000"],
            f"Обеспеченность на 10 000 населения: {region}",
            "Значение на 10 000 населения",
            labels={
                "doctors_per_10000": "Врачи",
                "nurses_per_10000": "Средний медперсонал",
                "beds_per_10000": "Больничные койки",
            },
        )
        line_chart_long(
            region_df,
            ["visits_per_person"],
            f"Посещения врачей на 1 человека: {region}",
            "Посещений на 1 человека",
            labels={"visits_per_person": "Посещения врачей"},
        )

    elif page == "Прогнозирование":
        region = st.selectbox("Регион", regions, index=default_region)
        horizon = st.slider("Горизонт прогноза, лет", 1, 5, 3)
        forecast_df = make_forecast(df, region, horizon, doctors_model, nurses_model, metadata)
        history = df[df["region"] == region].sort_values("year")

        st.subheader(f"Прогноз для региона: {region}")
        st.caption(
            "В версии 3 используется стабилизированный рекурсивный прогноз: выход ML-модели "
            "сопоставляется с устойчивой региональной динамикой последних лет, чтобы исключить "
            "нереалистичные скачки при прогнозе на несколько лет."
        )
        display_forecast_table(forecast_df)
        forecast_chart(
            history,
            forecast_df,
            ["doctors_count", "nurses_count"],
            f"Фактическая и прогнозная численность кадров: {region}",
        )
        forecast_chart(
            history,
            forecast_df,
            ["doctors_per_10000", "nurses_per_10000"],
            f"Фактическая и прогнозная обеспеченность кадрами: {region}",
        )

    elif page == "Оценка потребности":
        region = st.selectbox("Регион", regions, index=default_region)
        horizon = st.slider("Горизонт прогноза, лет", 1, 5, 5)
        doctors_norm = st.number_input(
            "Целевой норматив врачей на 10 000 населения",
            min_value=1.0,
            value=45.0,
            step=1.0,
        )
        nurses_norm = st.number_input(
            "Целевой норматив среднего медперсонала на 10 000 населения",
            min_value=1.0,
            value=95.0,
            step=1.0,
        )

        forecast_df = make_forecast(df, region, horizon, doctors_model, nurses_model, metadata)
        result = build_need_table(forecast_df, doctors_norm, nurses_norm)
        last = result.iloc[-1]

        st.subheader(f"Оценка кадровой потребности: {region}")
        st.caption(
            "Расчет потребности выполняется на основе стабилизированного прогноза численности кадров "
            "и выбранных пользователем нормативов обеспеченности на 10 000 населения."
        )
        col1, col2, col3, col4 = st.columns(4)
        col1.metric(f"Врачи, {int(last['year'])}", f"{last['doctors_count']:,.0f}".replace(",", " "))
        col2.metric("Дефицит/избыток врачей", f"{last['doctors_deficit']:,.0f}".replace(",", " "))
        col3.metric(f"Средний медперсонал, {int(last['year'])}", f"{last['nurses_count']:,.0f}".replace(",", " "))
        col4.metric("Дефицит/избыток среднего медперсонала", f"{last['nurses_deficit']:,.0f}".replace(",", " "))

        display_need_table(result)
        st.caption(
            "Положительное значение дефицита означает, что прогнозируемая численность кадров ниже расчетной потребности. "
            "Отрицательное значение означает избыток относительно выбранного норматива."
        )

    elif page == "Сравнение моделей":
        st.subheader("Качество моделей на отложенной временной выборке")
        metrics = pd.read_csv(REPORTS_PATH)
        metrics_display = metrics.copy()
        metrics_display["target"] = metrics_display["target"].map(TARGET_NAMES)
        metrics_display = metrics_display.round({"MAE": 2, "RMSE": 2, "R2": 4})
        show_dataframe(metrics_display.rename(columns=METRIC_COLUMNS), hide_index=True)

        fig = px.bar(
            metrics_display,
            x="model",
            y="MAE",
            color="target",
            barmode="group",
            labels={"model": "Модель", "MAE": "MAE", "target": "Целевой показатель"},
            title="Сравнение моделей по MAE",
        )
        fig.update_layout(legend_title_text="")
        show_plot(fig)

        st.write("Лучшие модели выбираются по минимальному MAE на тестовых годах.")
        best_models = pd.DataFrame(
            [
                {
                    "Целевой показатель": TARGET_NAMES[key],
                    "Лучшая модель": metadata["targets"][key]["best_model"],
                }
                for key in ["doctors", "nurses"]
            ]
        )
        show_dataframe(best_models, hide_index=True)


if __name__ == "__main__":
    main()
