"""
Feature-engineering parity test.

Compares the reusable build_features() implementation
against the already validated feature-engineered dataset.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.features.build_features import (
    build_features,
    get_feature_columns,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

REFERENCE_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "features"
    / "train.parquet"
)


def main():

    print("=" * 60)
    print("FEATURE ENGINEERING PARITY TEST")
    print("=" * 60)

    # Small deterministic sample for a fast comparison.
    reference_df = pd.read_parquet(
        REFERENCE_FILE
    ).head(1000).copy()

    print(f"\nReference shape: {reference_df.shape}")

    # Generate features using our reusable module.
    generated_df = build_features(
        reference_df.copy()
    )

    feature_columns = get_feature_columns()

    # --------------------------------------------------------------
    # 1. Check required columns
    # --------------------------------------------------------------

    missing = [
        col
        for col in feature_columns
        if col not in generated_df.columns
    ]

    assert not missing, (
        f"Generated features are missing: {missing}"
    )

    # --------------------------------------------------------------
    # 2. Compare feature values
    # --------------------------------------------------------------

    mismatches = []

    for column in feature_columns:

        reference = reference_df[column]
        generated = generated_df[column]

        # Numeric comparison allows small floating-point differences.
        if pd.api.types.is_numeric_dtype(reference):

            equal = np.allclose(
                reference.to_numpy(
                    dtype=float
                ),
                generated.to_numpy(
                    dtype=float
                ),
                equal_nan=True,
                rtol=1e-6,
                atol=1e-8,
            )

        else:

            equal = (
                reference.reset_index(drop=True)
                .equals(
                    generated.reset_index(drop=True)
                )
            )

        if not equal:
            mismatches.append(column)

    # --------------------------------------------------------------
    # 3. Final result
    # --------------------------------------------------------------

    if mismatches:

        print("\nFeature mismatches detected:")

        for column in mismatches:
            print(f"  - {column}")

        raise AssertionError(
            "Feature parity test failed."
        )

    print("\nAll model features match.")

    print(
        f"Checked {len(feature_columns)} "
        "model features."
    )

    print("\nFeature parity test passed.")


if __name__ == "__main__":
    main()