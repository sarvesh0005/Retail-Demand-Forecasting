from pathlib import Path

import pandas as pd


# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Data directory
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "m5"


def extract_data():
    """
    Read the raw M5 CSV files and return them as pandas DataFrames.
    """

    calendar_path = DATA_DIR / "calendar.csv"
    prices_path = DATA_DIR / "sell_prices.csv"
    sales_path = DATA_DIR / "sales_train_validation.csv"

    files = [calendar_path, prices_path, sales_path]

    # Validate files exist
    for file in files:
        if not file.exists():
            raise FileNotFoundError(f"File not found: {file}")

    # Read CSVs
    calendar_df = pd.read_csv(calendar_path)
    prices_df = pd.read_csv(prices_path)
    sales_df = pd.read_csv(sales_path)

    return calendar_df, prices_df, sales_df


if __name__ == "__main__":

    calendar_df, prices_df, sales_df = extract_data()

    print("Extraction Successful\n")

    print(f"Calendar : {calendar_df.shape}")
    print(f"Prices   : {prices_df.shape}")
    print(f"Sales    : {sales_df.shape}")