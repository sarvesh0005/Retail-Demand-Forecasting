import pandas as pd


def transform_data(calendar_df, prices_df, sales_df):
    """
    Transform raw M5 data into a model-ready transactional dataset.
    """

    # -----------------------------
    # Convert data types
    # -----------------------------
    calendar_df = calendar_df.copy()
    calendar_df["date"] = pd.to_datetime(calendar_df["date"])

    # -----------------------------
    # Convert Sales: Wide -> Long
    # -----------------------------
    id_cols = [
        "id",
        "item_id",
        "dept_id",
        "cat_id",
        "store_id",
        "state_id",
    ]

    day_cols = [col for col in sales_df.columns if col.startswith("d_")]

    sales_long = pd.melt(
        sales_df,
        id_vars=id_cols,
        value_vars=day_cols,
        var_name="d",
        value_name="sales",
    )

    # -----------------------------
    # Merge Calendar
    # -----------------------------
    sales_calendar = sales_long.merge(
        calendar_df,
        on="d",
        how="left",
    )

    # -----------------------------
    # Merge Sell Prices
    # -----------------------------
    sales_full = sales_calendar.merge(
        prices_df,
        on=["store_id", "item_id", "wm_yr_wk"],
        how="left",
    )

    # -----------------------------
    # Basic validation
    # -----------------------------
    assert sales_long.shape[0] == sales_calendar.shape[0]
    assert sales_calendar.shape[0] == sales_full.shape[0]

    return sales_full


if __name__ == "__main__":

    from src.etl.extract import extract_data

    calendar_df, prices_df, sales_df = extract_data()

    sales_full = transform_data(
        calendar_df,
        prices_df,
        sales_df,
    )

    print("Transformation Successful\n")
    print(f"Final Shape : {sales_full.shape}")
    print(sales_full.head())