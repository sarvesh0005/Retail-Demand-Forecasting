import pandas as pd


def transform_data(calendar_df, prices_df, sales_df):
    """
    Transform raw M5 data into normalized tables
    ready to be loaded into PostgreSQL.
    """

    # -----------------------------
    # Calendar Dimension
    # -----------------------------
    calendar_df = calendar_df.copy()
    calendar_df["date"] = pd.to_datetime(calendar_df["date"])

    # -----------------------------
    # Products Dimension
    # -----------------------------
    products_df = (
        sales_df[
            [
                "item_id",
                "dept_id",
                "cat_id",
            ]
        ]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    # -----------------------------
    # Stores Dimension
    # -----------------------------
    stores_df = (
        sales_df[
            [
                "store_id",
                "state_id",
            ]
        ]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    # -----------------------------
    # Prices Fact
    # -----------------------------
    prices_df = prices_df.copy()

    # -----------------------------
    # Sales Fact (Wide → Long)
    # -----------------------------
    id_columns = [
        "item_id",
        "store_id",
    ]

    day_columns = [
        col
        for col in sales_df.columns
        if col.startswith("d_")
    ]

    sales_long = pd.melt(
        sales_df,
        id_vars=id_columns,
        value_vars=day_columns,
        var_name="d",
        value_name="sales_quantity",
    )

    # Keep only columns present in database schema
    sales_long = sales_long[
        [
            "item_id",
            "store_id",
            "d",
            "sales_quantity",
        ]
    ]

    return (
        calendar_df,
        products_df,
        stores_df,
        prices_df,
        sales_long,
    )


if __name__ == "__main__":

    from src.etl.extract import extract_data

    calendar_df, prices_df, sales_df = extract_data()

    (
        calendar_df,
        products_df,
        stores_df,
        prices_df,
        sales_long,
    ) = transform_data(
        calendar_df,
        prices_df,
        sales_df,
    )

    print("Transformation Successful\n")

    print(f"Calendar : {calendar_df.shape}")
    print(f"Products : {products_df.shape}")
    print(f"Stores   : {stores_df.shape}")
    print(f"Prices   : {prices_df.shape}")
    print(f"Sales    : {sales_long.shape}")