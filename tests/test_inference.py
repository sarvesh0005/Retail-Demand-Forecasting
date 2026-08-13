"""
Inference sanity test.

Verifies that:
1. Saved model and preprocessor can be loaded.
2. Feature-engineered test data can be transformed.
3. Predictions can be generated successfully.
"""

from pathlib import Path

import pandas as pd

from src.inference.predict import predict_demand


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TEST_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "features"
    / "test.parquet"
)


def main():

    print("=" * 60)
    print("INFERENCE SANITY TEST")
    print("=" * 60)

    # Load a small sample only.
    test_df = pd.read_parquet(
        TEST_FILE
    ).head(10)

    print(f"\nInput rows: {len(test_df)}")

    # Generate predictions using saved artifacts.
    predictions = predict_demand(
        test_df
    )

    print("\nPredictions:")
    print(predictions)

    # Basic validation.
    assert len(predictions) == len(test_df)

    assert (
        "predicted_sales_quantity"
        in predictions.columns
    )

    assert (
        predictions["predicted_sales_quantity"]
        .notna()
        .all()
    )

    print("\nInference sanity test passed.")


if __name__ == "__main__":
    main()