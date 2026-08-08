from sqlalchemy import text

from src.database.connection import engine
from src.etl.extract import extract_data
from src.etl.transform import transform_data

CHUNK_SIZE = 10000


def load_table(df, table_name):
    """
    Load dataframe into PostgreSQL in chunks.
    """

    total_rows = len(df)

    print(f"\nLoading {table_name} ({total_rows:,} rows)...")

    for start in range(0, total_rows, CHUNK_SIZE):

        end = min(start + CHUNK_SIZE, total_rows)

        chunk = df.iloc[start:end]

        chunk.to_sql(
            table_name,
            con=engine,
            if_exists="append",
            index=False,
            method="multi",
        )

        print(f"Loaded rows {end:,}/{total_rows:,}")

    print(f"{table_name} loaded successfully.")


def load_data():

    (
        calendar_df,
        products_df,
        stores_df,
        prices_df,
        sales_df,
    ) = transform_data(*extract_data())

    # ==================================================
    # FIX: Convert integer SNAP columns to boolean
    # ==================================================

    bool_columns = [
        "snap_CA",
        "snap_TX",
        "snap_WI",
    ]

    calendar_df[bool_columns] = (
        calendar_df[bool_columns]
        .astype(bool)
    )

    # ==================================================
    # Clear existing data
    # ==================================================

    with engine.begin() as conn:

        conn.execute(text("TRUNCATE TABLE sales CASCADE"))
        conn.execute(text("TRUNCATE TABLE prices CASCADE"))
        conn.execute(text("TRUNCATE TABLE stores CASCADE"))
        conn.execute(text("TRUNCATE TABLE products CASCADE"))
        conn.execute(text("TRUNCATE TABLE calendar CASCADE"))

    # ==================================================
    # Load tables
    # ==================================================

    load_table(calendar_df, "calendar")
    load_table(products_df, "products")
    load_table(stores_df, "stores")
    load_table(prices_df, "prices")
    load_table(sales_df, "sales")


def verify_load():

    tables = [
        "calendar",
        "products",
        "stores",
        "prices",
        "sales",
    ]

    print("\nRow Counts\n")

    with engine.connect() as conn:

        for table in tables:

            count = conn.execute(
                text(f"SELECT COUNT(*) FROM {table}")
            ).scalar()

            print(f"{table:<10}: {count:,}")


if __name__ == "__main__":

    load_data()

    verify_load()