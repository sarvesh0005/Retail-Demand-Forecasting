"""
Preprocessing utilities for the retail demand forecasting model.

Responsibilities:
- Identify categorical and numerical features
- Build the preprocessing pipeline
- Fit the preprocessor on training data only
- Transform train/validation/test data
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


TARGET_COL = "sales_quantity"

IDENTIFIER_COLS = [
    "item_id",
    "store_id",
    "d",
    "date",
]


def get_feature_columns(
    X_train: pd.DataFrame,
) -> Tuple[List[str], List[str]]:
    """
    Identify categorical and numerical/binary feature columns.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training feature dataframe.

    Returns
    -------
    categorical_cols : list[str]
        Categorical feature columns.

    numerical_cols : list[str]
        Numerical and boolean feature columns.
    """

    categorical_cols = X_train.select_dtypes(
        include=["object", "category", "string"]
    ).columns.tolist()

    numerical_cols = X_train.select_dtypes(
        include=[np.number, "bool"]
    ).columns.tolist()

    unclassified = (
        set(X_train.columns)
        - set(categorical_cols)
        - set(numerical_cols)
    )

    if unclassified:
        raise ValueError(
            "Columns with unexpected dtypes: "
            f"{sorted(unclassified)}"
        )

    return categorical_cols, numerical_cols


def build_preprocessor(
    categorical_cols: List[str],
    numerical_cols: List[str],
) -> ColumnTransformer:
    """
    Build the preprocessing pipeline.

    Categorical features:
        OneHotEncoder(handle_unknown='ignore')

    Numerical/binary features:
        Passed through unchanged.

    Scaling is intentionally not used because the downstream
    model is XGBoost.
    """

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                categorical_cols,
            ),
            (
                "numerical",
                "passthrough",
                numerical_cols,
            ),
        ],
        remainder="drop",
    )

    return preprocessor


def fit_preprocessor(
    X_train: pd.DataFrame,
) -> tuple[
    ColumnTransformer,
    List[str],
    List[str],
]:
    """
    Build and fit the preprocessor using training data only.

    This is important for preventing preprocessing leakage.
    """

    categorical_cols, numerical_cols = get_feature_columns(
        X_train
    )

    preprocessor = build_preprocessor(
        categorical_cols=categorical_cols,
        numerical_cols=numerical_cols,
    )

    preprocessor.fit(X_train)

    return (
        preprocessor,
        categorical_cols,
        numerical_cols,
    )


def transform_data(
    preprocessor: ColumnTransformer,
    X: pd.DataFrame,
) -> np.ndarray:
    """
    Transform data using an already-fitted preprocessor.

    The preprocessor must already have been fitted on training data.
    """

    return preprocessor.transform(X)


def prepare_features(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Separate model features from the target and identifiers.

    Returns
    -------
    X : pd.DataFrame
        Model features.

    y : pd.Series
        Target variable.
    """

    if TARGET_COL not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COL}' not found in dataframe."
        )

    X = df.drop(
        columns=IDENTIFIER_COLS + [TARGET_COL],
        errors="ignore",
    )

    y = df[TARGET_COL]

    return X, y