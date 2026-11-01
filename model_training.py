"""Train ML models for forecasting medical staff counts by Russian regions.

Run from the project root:
    python model_training.py
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

from src.features import add_ml_features, clean_feature_rows, get_feature_columns

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "processed" / "healthcare_dataset_model_ready.csv"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
MODELS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)

PROJECT_VERSION = "3.0"

TARGETS = {
    "doctors": "target_doctors_next_year",
    "nurses": "target_nurses_next_year",
}

CANDIDATE_MODELS = {
    "Ridge Regression": Ridge(alpha=1.0),
    "Random Forest": RandomForestRegressor(
        n_estimators=120,
        max_depth=10,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=1,
    ),
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=160,
        learning_rate=0.04,
        max_depth=3,
        min_samples_leaf=3,
        random_state=42,
    ),
    "Decision Tree": DecisionTreeRegressor(max_depth=8, min_samples_leaf=3, random_state=42),
    "KNN Regressor": KNeighborsRegressor(n_neighbors=5, weights="distance"),
}


def build_pipeline(model, numeric_features: list[str]) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("region", OneHotEncoder(handle_unknown="ignore"), ["region"]),
            ("numeric", StandardScaler(), numeric_features),
        ],
        remainder="drop",
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def evaluate(y_true, y_pred) -> dict[str, float]:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "R2": float(r2_score(y_true, y_pred)),
    }


def train_for_target(df_features: pd.DataFrame, target_key: str, target_col: str) -> tuple[Pipeline, dict]:
    model_df = clean_feature_rows(df_features, target_col)
    feature_cols = get_feature_columns(model_df)
    numeric_features = [col for col in feature_cols if col != "region"]

    # Temporal holdout: train on older years, test on the last two available feature years.
    train_df = model_df[model_df["year"] <= 2021].copy()
    test_df = model_df[model_df["year"] >= 2022].copy()

    X_train, y_train = train_df[feature_cols], train_df[target_col]
    X_test, y_test = test_df[feature_cols], test_df[target_col]

    metrics_rows = []

    for model_name, model in CANDIDATE_MODELS.items():
        pipeline = build_pipeline(model, numeric_features)
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        metrics = evaluate(y_test, predictions)
        metrics_rows.append(
            {
                "target": target_key,
                "model": model_name,
                "train_rows": int(len(train_df)),
                "test_rows": int(len(test_df)),
                **metrics,
            }
        )

    # Choose the model with the lowest MAE on the temporal holdout.
    best_row = min(metrics_rows, key=lambda row: row["MAE"])
    best_model_name = best_row["model"]

    # Refit the selected model on all rows that have a known next-year target.
    final_pipeline = build_pipeline(CANDIDATE_MODELS[best_model_name], numeric_features)
    final_pipeline.fit(model_df[feature_cols], model_df[target_col])

    metadata = {
        "target_key": target_key,
        "target_column": target_col,
        "best_model": best_model_name,
        "feature_columns": feature_cols,
        "numeric_features": numeric_features,
        "holdout_train_years": "<= 2021",
        "holdout_test_years": ">= 2022",
        "metrics": metrics_rows,
    }
    return final_pipeline, metadata


def train_all() -> pd.DataFrame:
    raw_df = pd.read_csv(DATA_PATH)
    df_features = add_ml_features(raw_df, add_targets=True)

    all_metrics = []
    metadata = {
        "project_version": PROJECT_VERSION,
        "sklearn_version": sklearn.__version__,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "data_path": str(DATA_PATH.relative_to(ROOT)),
        "targets": {},
    }

    for target_key, target_col in TARGETS.items():
        pipeline, target_metadata = train_for_target(df_features, target_key, target_col)
        joblib.dump(pipeline, MODELS_DIR / f"{target_key}_model.pkl")
        metadata["targets"][target_key] = target_metadata
        all_metrics.extend(target_metadata["metrics"])

    metrics_df = pd.DataFrame(all_metrics)
    metrics_df = metrics_df.sort_values(["target", "MAE"]).reset_index(drop=True)
    metrics_df.to_csv(REPORTS_DIR / "model_metrics.csv", index=False, encoding="utf-8-sig")

    with open(MODELS_DIR / "model_metadata.json", "w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)

    return metrics_df


def main() -> None:
    metrics_df = train_all()
    print("Training completed.")
    print(metrics_df.round({"MAE": 2, "RMSE": 2, "R2": 4}))
    print(f"Models saved to: {MODELS_DIR}")
    print(f"Metrics saved to: {REPORTS_DIR / 'model_metrics.csv'}")


if __name__ == "__main__":
    main()
