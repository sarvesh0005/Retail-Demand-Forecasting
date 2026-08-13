"""
Model training utilities for retail demand forecasting.

Responsibilities:
- Define the selected XGBoost configuration
- Train the XGBoost model
- Generate predictions
"""

from __future__ import annotations

import numpy as np
import xgboost as xgb


def build_xgb_model() -> xgb.XGBRegressor:
    """
    Build the selected baseline XGBoost model.

    This configuration was selected based on validation performance.
    """

    return xgb.XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.10,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
    )


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> xgb.XGBRegressor:
    """
    Train the selected XGBoost model on preprocessed training data.
    """

    model = build_xgb_model()

    model.fit(
        X_train,
        y_train,
    )

    return model


def predict(
    model: xgb.XGBRegressor,
    X: np.ndarray,
) -> np.ndarray:
    """
    Generate predictions using a trained XGBoost model.
    """

    return model.predict(X)