"""Feature engineering for healthcare workforce forecasting."""
from __future__ import annotations

import numpy as np
import pandas as pd

BASE_INDICATORS = [
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

TARGET_COLUMNS = ["target_doctors_next_year", "target_nurses_next_year"]

RUSSIAN_COLUMNS = {
    "region": "Регион",
    "year": "Год",
    "population": "Численность населения",
    "doctors_count": "Численность врачей",
    "nurses_count": "Численность среднего медперсонала",
    "visits_count": "Число посещений врачей",
    "hospital_beds": "Число больничных коек",
    "outpatient_capacity": "Мощность амбулаторно-поликлинических организаций",
    "doctors_per_10000": "Врачей на 10 000 населения",
    "nurses_per_10000": "Среднего медперсонала на 10 000 населения",
    "beds_per_10000": "Коек на 10 000 населения",
    "visits_per_person": "Посещений врачей на 1 человека",
    "doctors_need": "Расчетная потребность во врачах",
    "nurses_need": "Расчетная потребность в среднем медперсонале",
    "doctors_deficit": "Дефицит/избыток врачей",
    "nurses_deficit": "Дефицит/избыток среднего медперсонала",
    "doctors_status": "Статус по врачам",
    "nurses_status": "Статус по среднему медперсоналу",
}

METRIC_COLUMNS = {
    "target": "Целевой показатель",
    "model": "Модель",
    "train_rows": "Обучающая выборка",
    "test_rows": "Тестовая выборка",
    "MAE": "MAE",
    "RMSE": "RMSE",
    "R2": "R²",
}

TARGET_NAMES = {
    "doctors": "Численность врачей",
    "nurses": "Численность среднего медперсонала",
}


def add_ml_features(df: pd.DataFrame, add_targets: bool = True) -> pd.DataFrame:
    """Add lag and growth features by region.

    The model predicts next-year values using information available in the
    current and previous years. This avoids using future values in features.
    """
    result = df.copy()
    result = result.sort_values(["region", "year"]).reset_index(drop=True)

    for col in BASE_INDICATORS:
        grouped = result.groupby("region", sort=False)[col]
        result[f"{col}_lag_1"] = grouped.shift(1)
        result[f"{col}_lag_2"] = grouped.shift(2)
        result[f"{col}_growth_1"] = grouped.pct_change()
        result[f"{col}_growth_1"] = result[f"{col}_growth_1"].replace([np.inf, -np.inf], np.nan)

    if add_targets:
        result["target_doctors_next_year"] = result.groupby("region", sort=False)["doctors_count"].shift(-1)
        result["target_nurses_next_year"] = result.groupby("region", sort=False)["nurses_count"].shift(-1)

    return result


def get_feature_columns(df_with_features: pd.DataFrame) -> list[str]:
    """Return feature columns used by ML models."""
    excluded = set(TARGET_COLUMNS)
    return [col for col in df_with_features.columns if col not in excluded]


def clean_feature_rows(df_with_features: pd.DataFrame, target_column: str | None = None) -> pd.DataFrame:
    """Drop rows with missing model features and optionally target."""
    feature_cols = get_feature_columns(df_with_features)
    subset = feature_cols if target_column is None else feature_cols + [target_column]
    return df_with_features.dropna(subset=subset).copy()


def rename_for_display(df: pd.DataFrame) -> pd.DataFrame:
    """Rename known technical columns to Russian display names."""
    return df.rename(columns={col: RUSSIAN_COLUMNS.get(col, col) for col in df.columns})
