"""
Evaluation metrics for retail demand forecasting.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
)


def calculate_mae(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """Calculate Mean Absolute Error."""

    return float(
        mean_absolute_error(y_true, y_pred)
    )


def calculate_rmse(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """Calculate Root Mean Squared Error."""

    return float(
        np.sqrt(
            mean_squared_error(y_true, y_pred)
        )
    )


def calculate_wape(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """
    Calculate Weighted Absolute Percentage Error.

    WAPE = sum(|actual - prediction|)
          / sum(|actual|)
    """

    denominator = np.sum(np.abs(y_true))

    if denominator == 0:
        return float("nan")

    return float(
        np.sum(np.abs(y_true - y_pred))
        / denominator
    )


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """
    Calculate all primary forecasting metrics.
    """

    return {
        "mae": calculate_mae(y_true, y_pred),
        "rmse": calculate_rmse(y_true, y_pred),
        "wape": calculate_wape(y_true, y_pred),
    }