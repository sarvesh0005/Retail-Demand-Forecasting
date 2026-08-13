"""
Feature engineering for retail demand forecasting.

This module contains the validated feature logic from
05_feature_engineering.ipynb.

Responsibilities:
- Calendar features
- Price features
- Historical sales lag features
- Leakage-safe rolling features

The module does NOT:
- split train/validation/test data
- preprocess categorical variables
- train models
- impute lag/rolling NaNs
"""

from __future__ import annotations

import numpy as np
import pandas as pd


GROUP_KEYS = ["item_id", "store_id"]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the complete feature set used by the forecasting model.

    Parameters
    ----------
    df : pd.DataFrame
        Analytical dataset containing sales, price, date,
        and categorical/calendar information.

    Returns
    -------
    pd.DataFrame
        Feature-engineered dataframe.
    """

    required_columns = [
        "item_id",
        "store_id",
        "date",
        "sales_quantity",
        "sell_price",
        "event_name_1",
        "event_name_2",
        "month",
        "year",
    ]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    df = df.copy()

    # --------------------------------------------------------------
    # 1. Date preparation
    # --------------------------------------------------------------

    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"])

    # Sort before any lag/rolling operation.
    df = df.sort_values(
        GROUP_KEYS + ["date"]
    ).reset_index(drop=True)

    # --------------------------------------------------------------
    # 2. Calendar features
    # --------------------------------------------------------------

    df["day_of_week"] = df["date"].dt.dayofweek
    df["day_of_month"] = df["date"].dt.day
    df["week_of_year"] = (
        df["date"]
        .dt.isocalendar()
        .week
        .astype(int)
    )
    df["quarter"] = df["date"].dt.quarter

    # Binary event indicator.
    df["is_event"] = (
        df["event_name_1"].notna()
        | df["event_name_2"].notna()
    ).astype("int8")

    # --------------------------------------------------------------
    # 3. Price features
    # --------------------------------------------------------------

    df["price_lag_1"] = (
        df.groupby(GROUP_KEYS)["sell_price"]
        .shift(1)
    )

    df["price_change"] = (
        df["sell_price"]
        - df["price_lag_1"]
    )

    # Avoid division by zero.
    df["price_change_pct"] = np.where(
        df["price_lag_1"].fillna(0) == 0,
        np.nan,
        df["price_change"]
        / df["price_lag_1"],
    )

    # Same-day average price for the same item
    # across stores.
    same_day_item_avg_price = (
        df.groupby(
            ["date", "item_id"]
        )["sell_price"]
        .transform("mean")
    )

    df["relative_price"] = (
        df["sell_price"]
        / same_day_item_avg_price
    )

    # --------------------------------------------------------------
    # 4. Historical demand lag features
    # --------------------------------------------------------------

    df["sales_lag_1"] = (
        df.groupby(GROUP_KEYS)["sales_quantity"]
        .shift(1)
    )

    df["sales_lag_7"] = (
        df.groupby(GROUP_KEYS)["sales_quantity"]
        .shift(7)
    )

    df["sales_lag_28"] = (
        df.groupby(GROUP_KEYS)["sales_quantity"]
        .shift(28)
    )

    # --------------------------------------------------------------
    # 5. Leakage-safe rolling features
    # --------------------------------------------------------------

    # IMPORTANT:
    #
    # Shift FIRST so that the current target is excluded.
    #
    # rolling feature at t
    # = historical values t-1 ... t-window
    #
    shifted_sales = (
        df.groupby(GROUP_KEYS)["sales_quantity"]
        .shift(1)
    )

    grouped_shifted_sales = shifted_sales.groupby(
        [df["item_id"], df["store_id"]]
    )

    df["rolling_mean_7"] = (
        grouped_shifted_sales
        .transform(
            lambda s: s.rolling(7).mean()
        )
    )

    df["rolling_mean_28"] = (
        grouped_shifted_sales
        .transform(
            lambda s: s.rolling(28).mean()
        )
    )

    df["rolling_std_7"] = (
        grouped_shifted_sales
        .transform(
            lambda s: s.rolling(7).std()
        )
    )

    df["rolling_std_28"] = (
        grouped_shifted_sales
        .transform(
            lambda s: s.rolling(28).std()
        )
    )

    # --------------------------------------------------------------
    # 6. Validation
    # --------------------------------------------------------------

    _validate_features(df)

    return df


def _validate_features(df: pd.DataFrame) -> None:
    """
    Validate important properties of generated features.

    Raises
    ------
    ValueError
        If duplicates or infinite values are detected.
    """

    # No duplicate series-date observations.
    duplicate_count = df.duplicated(
        subset=[
            "item_id",
            "store_id",
            "date",
        ]
    ).sum()

    if duplicate_count != 0:
        raise ValueError(
            "Duplicate (item_id, store_id, date) "
            f"rows detected: {duplicate_count}"
        )

    # Check numerical columns for infinity.
    numeric_cols = df.select_dtypes(
        include=[np.number]
    ).columns

    inf_count = np.isinf(
        df[numeric_cols]
    ).sum().sum()

    if inf_count != 0:
        raise ValueError(
            f"Infinite values detected: {inf_count}"
        )


def get_feature_columns() -> list[str]:
    """
    Return the validated model feature list.

    This matches the feature set used by the trained
    XGBoost model.
    """

    return [
        # Calendar
        "day_of_week",
        "day_of_month",
        "week_of_year",
        "month",
        "quarter",
        "year",

        # Events
        "is_event",

        # SNAP
        "snap_CA",
        "snap_TX",
        "snap_WI",

        # Price
        "sell_price",
        "price_lag_1",
        "price_change",
        "price_change_pct",
        "relative_price",

        # Historical demand
        "sales_lag_1",
        "sales_lag_7",
        "sales_lag_28",
        "rolling_mean_7",
        "rolling_mean_28",
        "rolling_std_7",
        "rolling_std_28",

        # Categorical dimensions
        "weekday",
        "dept_id",
        "cat_id",
        "state_id",
    ]