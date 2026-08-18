"""
End-to-end inference sanity test.

Analytical data
    ↓
Feature engineering
    ↓
Preprocessing
    ↓
XGBoost
    ↓
Prediction
"""

from pathlib import Path

import pandas as pd

from src.inference.predict import predict_demand


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "training_dataset"
    / "train_dataset.parquet"
)


def main():

    print("=" * 60)
    print("END-TO-END INFERENCE TEST")
    print("=" * 60)

    # Load analytical data.
    df = pd.read_parquet(
        INPUT_FILE
    )

    # Use a complete historical period rather than
    # randomly selecting rows, because lag/rolling
    # features require temporal history.
    df = df.sort_values(
        ["item_id", "store_id", "date"]
    )

    # Keep the first 60 days.
    selected_dates = sorted(
        df["date"].unique()
    )[:60]

    sample = df[
        df["date"].isin(selected_dates)
    ].copy()

    print(
        f"\nInput rows: {len(sample):,}"
    )

    # --------------------------------------------------------------
    # Run complete inference pipeline
    # --------------------------------------------------------------

    predictions = predict_demand(
        sample
    )

    print(
        f"Prediction rows: "
        f"{len(predictions):,}"
    )

    print("\nSample predictions:")
    print(
        predictions.head(10)
    )

    # --------------------------------------------------------------
    # Basic validation
    # --------------------------------------------------------------

    assert len(predictions) == len(sample)

    assert (
        "predicted_sales_quantity"
        in predictions.columns
    )

    assert (
        predictions[
            "predicted_sales_quantity"
        ]
        .notna()
        .all()
    )

    print(
        "\nEnd-to-end inference test passed."
    )


if __name__ == "__main__":
    main()