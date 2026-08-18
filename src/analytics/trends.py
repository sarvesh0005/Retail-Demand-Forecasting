"""
Demand trend analytics for the Retail Demand Intelligence dashboard.
"""

from datetime import date

import pandas as pd

from src.analytics.summary import load_analytical_data


def get_demand_trend(
    store_id: str | None = None,
    cat_id: str | None = None,
    item_id: str | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
) -> list[dict]:
    """
    Return daily aggregated demand with optional filters.
    """

    df = load_analytical_data()

    # --------------------------------------------------------------
    # Apply filters
    # --------------------------------------------------------------

    if store_id is not None:
        df = df[
            df["store_id"] == store_id
        ]

    if cat_id is not None:
        df = df[
            df["cat_id"] == cat_id
        ]

    if item_id is not None:
        df = df[
            df["item_id"] == item_id
        ]

    if start_date is not None:
        df = df[
            df["date"] >= pd.Timestamp(
                start_date
            )
        ]

    if end_date is not None:
        df = df[
            df["date"] <= pd.Timestamp(
                end_date
            )
        ]

    # --------------------------------------------------------------
    # Validate filtered data
    # --------------------------------------------------------------

    if df.empty:
        return []

    # --------------------------------------------------------------
    # Aggregate daily demand
    # --------------------------------------------------------------

    trend = (
        df.groupby("date", as_index=False)
        ["sales_quantity"]
        .sum()
        .sort_values("date")
    )

    # Convert to API-friendly format.
    trend["date"] = (
        trend["date"]
        .dt.strftime("%Y-%m-%d")
    )

    trend["sales_quantity"] = (
        trend["sales_quantity"]
        .astype(float)
    )

    return trend.to_dict(
        orient="records"
    )