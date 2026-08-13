"""
Feature engineering sanity test.
"""

from pathlib import Path

import pandas as pd

from src.features.build_features import (
    build_features,
    get_feature_columns,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "features"
    / "train.parquet"
)


def main():

    print("=" * 60)
    print("FEATURE ENGINEERING SANITY TEST")
    print("=" * 60)

    # Use a small sample for a fast test.
    df = pd.read_parquet(INPUT_FILE).head(1000)

    print(f"\nInput shape: {df.shape}")

    features = build_features(df)

    print(f"Output shape: {features.shape}")

    feature_columns = get_feature_columns()

    missing = [
        col
        for col in feature_columns
        if col not in features.columns
    ]

    assert not missing, (
        f"Missing expected features: {missing}"
    )

    assert not features[
        ["item_id", "store_id", "date"]
    ].duplicated().any()

    print(
        f"Expected model features: "
        f"{len(feature_columns)}"
    )

    print("\nFeature engineering test passed.")


if __name__ == "__main__":
    main()