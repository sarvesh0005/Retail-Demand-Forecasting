"""
Inference utilities for retail demand forecasting.

Inference flow:

Analytical data
    ↓
Feature engineering
    ↓
Saved preprocessor
    ↓
Saved XGBoost model
    ↓
Demand prediction
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from src.features.build_features import build_features
from src.preprocessing.preprocessor import prepare_features
from src.models.train import predict


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_FILE = (
    PROJECT_ROOT
    / "artifacts"
    / "model"
    / "xgb_model.pkl"
)

PREPROCESSOR_FILE = (
    PROJECT_ROOT
    / "artifacts"
    / "preprocessing"
    / "preprocessor.pkl"
)


def load_artifacts():
    """Load the trained model and fitted preprocessor."""

    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Model artifact not found: {MODEL_FILE}"
        )

    if not PREPROCESSOR_FILE.exists():
        raise FileNotFoundError(
            f"Preprocessor artifact not found: "
            f"{PREPROCESSOR_FILE}"
        )

    model = joblib.load(MODEL_FILE)
    preprocessor = joblib.load(
        PREPROCESSOR_FILE
    )

    return model, preprocessor


def predict_demand(
    analytical_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Generate demand predictions from analytical data.

    The input must contain sufficient historical data for
    lag and rolling features to be calculated correctly.

    Parameters
    ----------
    analytical_df : pd.DataFrame
        Analytical dataset before feature engineering.

    Returns
    -------
    pd.DataFrame
        Identifier columns and predicted demand.
    """

    if analytical_df.empty:
        raise ValueError(
            "Input dataframe is empty."
        )

    # --------------------------------------------------------------
    # 1. Feature engineering
    # --------------------------------------------------------------

    feature_df = build_features(
        analytical_df.copy()
    )

    # --------------------------------------------------------------
    # 2. Load trained artifacts
    # --------------------------------------------------------------

    model, preprocessor = load_artifacts()

    # --------------------------------------------------------------
    # 3. Prepare model features
    # --------------------------------------------------------------

    X, _ = prepare_features(
        feature_df
    )

    # --------------------------------------------------------------
    # 4. Apply fitted preprocessing
    # --------------------------------------------------------------

    X_processed = preprocessor.transform(
        X
    )

    # --------------------------------------------------------------
    # 5. Generate predictions
    # --------------------------------------------------------------

    predictions = predict(
        model,
        X_processed
    )

    # --------------------------------------------------------------
    # 6. Return predictions
    # --------------------------------------------------------------

    result = feature_df[
        [
            "item_id",
            "store_id",
            "date",
        ]
    ].copy()

    result[
        "predicted_sales_quantity"
    ] = predictions

    return result