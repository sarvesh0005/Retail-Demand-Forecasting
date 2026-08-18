"""
Feature-engineering parity test.

Generates features from the pre-feature-engineering analytical
dataset and compares them against the validated feature dataset.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.features.build_features import (
    build_features,
    get_feature_columns,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_ANALYTICAL_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "training_dataset"
    / "train_dataset.parquet"
)

REFERENCE_FEATURE_FILE = (
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

    # --------------------------------------------------------------
    # 1. Load datasets
    # --------------------------------------------------------------

    raw_df = pd.read_parquet(
        RAW_ANALYTICAL_FILE
    )

    reference_df = pd.read_parquet(
        REFERENCE_FEATURE_FILE
    )

    # Normalize date representation.
    # Raw analytical data stores date as object/string,
    # while Parquet feature data stores it as datetime64.
    raw_df["date"] = pd.to_datetime(
        raw_df["date"]
    )

    reference_df["date"] = pd.to_datetime(
        reference_df["date"]
    )

    print(
        f"\nAnalytical dataset : "
        f"{raw_df.shape}"
    )

    print(
        f"Reference features : "
        f"{reference_df.shape}"
    )

    # --------------------------------------------------------------
    # 2. Select dates that exist in the reference dataset
    # --------------------------------------------------------------

    reference_dates = sorted(
        reference_df["date"].unique()
    )

    selected_dates = reference_dates[:15]

    raw_sample = raw_df[
        raw_df["date"].isin(selected_dates)
    ].copy()

    reference_sample = reference_df[
        reference_df["date"].isin(selected_dates)
    ].copy()

    print(
        f"\nSelected dates     : "
        f"{selected_dates[0]} → "
        f"{selected_dates[-1]}"
    )

    print(
        f"Analytical sample  : "
        f"{len(raw_sample):,} rows"
    )

    print(
        f"Reference sample   : "
        f"{len(reference_sample):,} rows"
    )

    # --------------------------------------------------------------
    # 3. Basic sanity check
    # --------------------------------------------------------------

    if raw_sample.empty:
        raise AssertionError(
            "Analytical sample is empty."
        )

    if reference_sample.empty:
        raise AssertionError(
            "Reference feature sample is empty."
        )

    # --------------------------------------------------------------
    # 4. Generate features
    # --------------------------------------------------------------

    print("\nGenerating features...")

    generated_df = build_features(
        raw_sample
    )

    print(
        f"Generated features : "
        f"{generated_df.shape}"
    )

    # --------------------------------------------------------------
    # 5. Align row ordering
    # --------------------------------------------------------------

    key_columns = [
        "item_id",
        "store_id",
        "date",
    ]

    generated_df = (
        generated_df
        .sort_values(key_columns)
        .reset_index(drop=True)
    )

    reference_sample = (
        reference_sample
        .sort_values(key_columns)
        .reset_index(drop=True)
    )

    # --------------------------------------------------------------
    # 6. Check row counts
    # --------------------------------------------------------------

    if len(generated_df) != len(reference_sample):

        raise AssertionError(
            "Row count mismatch: "
            f"{len(generated_df)} generated vs "
            f"{len(reference_sample)} reference."
        )

    # --------------------------------------------------------------
# 7. Check identifier alignment
# --------------------------------------------------------------

# Compare identifier values after normalizing their
# representation. This avoids false failures caused by
# different datetime resolutions such as datetime64[ns]
# versus datetime64[ms].

    for column in key_columns:

        generated_values = (
            generated_df[column]
            .astype("string")
            .reset_index(drop=True)
        )

        reference_values = (
            reference_sample[column]
            .astype("string")
            .reset_index(drop=True)
        )

        if not generated_values.equals(
            reference_values
        ):

            raise AssertionError(
                f"Identifier ordering mismatch "
                f"in column: {column}"
            )

    # --------------------------------------------------------------
    # 8. Compare model features
    # --------------------------------------------------------------

    feature_columns = get_feature_columns()

    print(
        f"\nComparing "
        f"{len(feature_columns)} model features..."
    )

    mismatches = []

    for column in feature_columns:

        generated = generated_df[column]
        reference = reference_sample[column]

        # ----------------------------------------------------------
        # Numeric features
        # ----------------------------------------------------------

        if (
            pd.api.types.is_numeric_dtype(
                generated
            )
            and pd.api.types.is_numeric_dtype(
                reference
            )
        ):

            equal = np.allclose(
                generated.to_numpy(
                    dtype=float
                ),
                reference.to_numpy(
                    dtype=float
                ),
                equal_nan=True,
                rtol=1e-6,
                atol=1e-8,
            )

        # ----------------------------------------------------------
        # Non-numeric / categorical features
        # ----------------------------------------------------------

        else:

            equal = (
                generated
                .astype("string")
                .reset_index(drop=True)
                .equals(
                    reference
                    .astype("string")
                    .reset_index(drop=True)
                )
            )

        if not equal:
            mismatches.append(column)

    # --------------------------------------------------------------
    # 9. Final result
    # --------------------------------------------------------------

    if mismatches:

        print(
            "\nFeature mismatches detected:"
        )

        for column in mismatches:
            print(f"  - {column}")

        raise AssertionError(
            "\nFeature parity test failed."
        )

    print(
        f"\nAll {len(feature_columns)} "
        "model features match."
    )

    print(
        "\nFeature parity test passed."
    )


if __name__ == "__main__":
    main()