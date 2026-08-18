"""
Analytics utilities for the Retail Demand Intelligence dashboard.
"""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "training_dataset"
    / "train_dataset.parquet"
)


def load_analytical_data() -> pd.DataFrame:
    """Load the analytical dataset used by the dashboard."""

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Analytical dataset not found: {DATA_FILE}"
        )

    df = pd.read_parquet(DATA_FILE)

    df["date"] = pd.to_datetime(df["date"])

    return df


def get_summary() -> dict:
    """
    Calculate high-level retail demand statistics.
    """

    df = load_analytical_data()

    total_sales = float(
        df["sales_quantity"].sum()
    )

    average_daily_demand = float(
        df.groupby("date")["sales_quantity"]
        .sum()
        .mean()
    )

    category_sales = (
        df.groupby("cat_id")["sales_quantity"]
        .sum()
        .sort_values(ascending=False)
    )

    store_sales = (
        df.groupby("store_id")["sales_quantity"]
        .sum()
        .sort_values(ascending=False)
    )

    return {
        "total_sales": total_sales,
        "average_daily_demand": average_daily_demand,
        "top_category": category_sales.index[0],
        "top_store": store_sales.index[0],
    }