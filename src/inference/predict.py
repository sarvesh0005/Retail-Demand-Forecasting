"""
Inference utilities for retail demand forecasting.

Loads the persisted preprocessing pipeline and XGBoost model,
then generates predictions for new feature-engineered data.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

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
            f"Preprocessor artifact not found: {PREPROCESSOR_FILE}"
        )

    model = joblib.load(MODEL_FILE)
    preprocessor = joblib.load(PREPROCESSOR_FILE)

    return model, preprocessor


def predict_demand(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Generate demand predictions for feature-engineered data.

    Parameters
    ----------
    df : pd.DataFrame
        Feature-engineered observations.

    Returns
    -------
    pd.DataFrame
        Original identifier columns with predictions.
    """

    model, preprocessor = load_artifacts()

    X, _ = prepare_features(df)

    X_processed = preprocessor.transform(X)

    predictions = predict(
        model,
        X_processed,
    )

    result = df[
        ["item_id", "store_id", "date"]
    ].copy()

    result["predicted_sales_quantity"] = predictions

    return result