"""
FastAPI prediction integration test.

Tests the complete API flow:

Analytical data
    ↓
JSON serialization
    ↓
POST /predict
    ↓
Pydantic validation
    ↓
Feature engineering
    ↓
Preprocessing
    ↓
XGBoost
    ↓
JSON response
"""

from pathlib import Path

import pandas as pd
import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "training_dataset"
    / "train_dataset.parquet"
)

API_URL = "http://127.0.0.1:8000/predict"


def main():

    print("=" * 60)
    print("FASTAPI PREDICTION TEST")
    print("=" * 60)

    # --------------------------------------------------------------
    # 1. Load analytical data
    # --------------------------------------------------------------

    df = pd.read_parquet(
        INPUT_FILE
    )

    # Use one complete item-store time series.
    # This preserves the historical context required
    # for lag and rolling features.
    df = df[
        (df["item_id"] == "FOODS_1_001")
        & (df["store_id"] == "CA_1")
    ].copy()

    df = df.sort_values("date")

    # Keep sufficient historical observations.
    df = df.head(60)

    print(
        f"\nInput rows: {len(df)}"
    )

    # --------------------------------------------------------------
    # 2. Prepare data for JSON
    # --------------------------------------------------------------

    # Convert date to JSON-compatible string.
    df["date"] = pd.to_datetime(
        df["date"]
    ).dt.strftime("%Y-%m-%d")

    # Pandas uses NaN for missing values.
    # Standard JSON does not support NaN.
    #
    # Convert missing values to Python None,
    # which becomes JSON null.
    df = df.astype(object).where(
        pd.notna(df),
        None
    )

    payload = {
        "data": df.to_dict(
            orient="records"
        )
    }

    # --------------------------------------------------------------
    # 3. Send request
    # --------------------------------------------------------------

    print(
        "\nSending request to API..."
    )

    response = requests.post(
        API_URL,
        json=payload,
        timeout=120,
    )

    print(
        f"HTTP status: "
        f"{response.status_code}"
    )

    # Print API error body if request failed.
    if not response.ok:
        print(
            "\nAPI error response:"
        )
        print(
            response.text
        )

    response.raise_for_status()

    # --------------------------------------------------------------
    # 4. Parse response
    # --------------------------------------------------------------

    result = response.json()

    print(
        f"Predictions returned: "
        f"{len(result)}"
    )

    # --------------------------------------------------------------
    # 5. Validate response
    # --------------------------------------------------------------

    assert isinstance(
        result,
        list
    )

    assert len(result) == len(df)

    required_fields = {
        "item_id",
        "store_id",
        "date",
        "predicted_sales_quantity",
    }

    assert required_fields.issubset(
        result[0].keys()
    )

    # Make sure predictions are actually populated.
    for row in result:

        assert (
            row["predicted_sales_quantity"]
            is not None
        )

    # --------------------------------------------------------------
    # 6. Display sample
    # --------------------------------------------------------------

    print(
        "\nSample API response:"
    )

    for row in result[:5]:
        print(row)

    print(
        "\nFastAPI prediction test passed."
    )


if __name__ == "__main__":
    main()