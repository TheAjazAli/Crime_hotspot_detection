"""
Core ML pipeline for the Crime Hotspot Prediction & Visualization Dashboard.

Approach
--------
1. The city/region is divided into a spatial grid (lat/lon rounded to a
   configurable precision). Each grid cell's total crime count is computed.
2. Cells in the top `hotspot_quantile` of crime density are labeled
   HOTSPOT = 1, the rest HOTSPOT = 0. Every record inherits its cell's label.
3. Four classifiers are trained to predict the HOTSPOT label from
   non-count features (crime type, time of day, day of week, month,
   weapon, etc.) so the model learns *patterns* rather than memorizing
   the count itself:
      - Decision Tree
      - Random Forest
      - XGBoost
      - Ensemble (soft-voting combination of the three)
4. All four are evaluated on a held-out test set (accuracy, precision,
   recall, F1, ROC-AUC, confusion matrix) so they can be compared head
   to head — this is the core empirical result for the research paper.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix,
)
from xgboost import XGBClassifier

REQUIRED_COLUMNS = ["latitude", "longitude"]
OPTIONAL_FEATURE_COLUMNS = [
    "crime_type", "hour", "weapon", "victim_age", "status", "date",
]


class DatasetValidationError(Exception):
    pass


def validate_dataset(df: pd.DataFrame) -> list[str]:
    """Returns a list of human-readable warnings; raises if unusable."""
    warnings = []
    missing_required = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_required:
        raise DatasetValidationError(
            f"Missing required column(s): {', '.join(missing_required)}. "
            f"The dataset must include latitude and longitude."
        )
    if df["latitude"].isna().all() or df["longitude"].isna().all():
        raise DatasetValidationError("latitude/longitude columns are empty.")

    present_optional = [c for c in OPTIONAL_FEATURE_COLUMNS if c in df.columns]
    if not present_optional:
        warnings.append(
            "No optional columns (crime_type, hour, weapon, victim_age, status, date) "
            "found — predictions will rely on location and count alone, which is weak signal."
        )
    return warnings


def engineer_features(df: pd.DataFrame, grid_precision: float = 0.01,
                       hotspot_quantile: float = 0.75):
    """
    Bins records into a spatial grid, labels hotspot cells, and builds
    a model-ready feature matrix.

    grid_precision: size of each grid cell in degrees (~0.01 ≈ 1.1 km)
    hotspot_quantile: cells at/above this crime-count quantile = hotspot
    """
    df = df.copy()
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df = df.dropna(subset=["latitude", "longitude"])

    # Spatial grid cell
    df["grid_lat"] = (df["latitude"] / grid_precision).round() * grid_precision
    df["grid_lon"] = (df["longitude"] / grid_precision).round() * grid_precision
    df["cell_id"] = df["grid_lat"].astype(str) + "_" + df["grid_lon"].astype(str)

    # Crime count per cell -> hotspot label
    cell_counts = df.groupby("cell_id").size().rename("cell_crime_count")
    df = df.merge(cell_counts, on="cell_id", how="left")
    threshold = df["cell_crime_count"].quantile(hotspot_quantile)
    df["hotspot"] = (df["cell_crime_count"] >= threshold).astype(int)

    # Temporal features
    if "date" in df.columns:
        parsed = pd.to_datetime(df["date"], errors="coerce")
        df["month"] = parsed.dt.month.fillna(0).astype(int)
        df["day_of_week"] = parsed.dt.dayofweek.fillna(0).astype(int)
    else:
        df["month"], df["day_of_week"] = 0, 0

    if "hour" in df.columns:
        df["hour"] = pd.to_numeric(df["hour"], errors="coerce").fillna(-1).astype(int)
    else:
        df["hour"] = -1

    # Encode categoricals. Note: grid_lat/grid_lon are deliberately EXCLUDED
    # from the feature set — since the hotspot label is derived directly from
    # each grid cell's count, including the cell's own coordinates would let
    # a model trivially memorize the label instead of learning real patterns.
    # The question the models actually answer is: "do a crime's characteristics
    # (type, timing, weapon, etc.) predict whether it occurred in a hotspot?"
    encoders: dict[str, LabelEncoder] = {}
    feature_cols = ["month", "day_of_week", "hour"]

    for cat_col in ["crime_type", "weapon", "status"]:
        if cat_col in df.columns:
            le = LabelEncoder()
            df[f"{cat_col}_enc"] = le.fit_transform(df[cat_col].astype(str))
            encoders[cat_col] = le
            feature_cols.append(f"{cat_col}_enc")

    if "victim_age" in df.columns:
        df["victim_age"] = pd.to_numeric(df["victim_age"], errors="coerce").fillna(
            df["victim_age"].median() if df["victim_age"].notna().any() else 30
        )
        feature_cols.append("victim_age")

    meta = {
        "feature_cols": feature_cols,
        "encoders": encoders,
        "grid_precision": grid_precision,
        "hotspot_quantile": hotspot_quantile,
        "threshold_count": float(threshold),
    }
    return df, meta


def train_and_compare(df: pd.DataFrame, meta: dict, test_size: float = 0.25,
                       random_state: int = 42):
    """Trains all four models and returns metrics + fitted models + test predictions."""
    X = df[meta["feature_cols"]]
    y = df["hotspot"]

    if y.nunique() < 2:
        raise DatasetValidationError(
            "Only one hotspot class present after labeling — try a different "
            "hotspot quantile or check your location spread."
        )

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, df.index, test_size=test_size, random_state=random_state, stratify=y
    )

    models = {
        "Decision Tree": DecisionTreeClassifier(max_depth=8, random_state=random_state),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=10, random_state=random_state, n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            eval_metric="logloss", random_state=random_state, n_jobs=-1
        ),
    }
    ensemble = VotingClassifier(
        estimators=[(k, v) for k, v in models.items()], voting="soft"
    )
    models["Ensemble (Voting)"] = ensemble

    results = {}
    fitted = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]

        results[name] = {
            "accuracy": accuracy_score(y_test, preds),
            "precision": precision_score(y_test, preds, zero_division=0),
            "recall": recall_score(y_test, preds, zero_division=0),
            "f1": f1_score(y_test, preds, zero_division=0),
            "roc_auc": roc_auc_score(y_test, probs),
            "confusion_matrix": confusion_matrix(y_test, preds).tolist(),
        }
        fitted[name] = model

    best_model_name = max(results, key=lambda k: results[k]["f1"])

    # Predict on the full dataset with the best model, for map visualization
    df_out = df.copy()
    df_out["predicted_hotspot_prob"] = fitted[best_model_name].predict_proba(X)[:, 1]
    df_out["predicted_hotspot"] = fitted[best_model_name].predict(X)

    return {
        "results": results,
        "fitted_models": fitted,
        "best_model_name": best_model_name,
        "df_predictions": df_out,
        "test_index": idx_test,
    }
