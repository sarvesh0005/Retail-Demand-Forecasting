"""
End-to-end model training runner.

Flow:
    Feature Parquet files
        ↓
    Feature / target separation
        ↓
    Fit preprocessing on train only
        ↓
    Transform train / validation / test
        ↓
    Train XGBoost
        ↓
    Evaluate validation
        ↓
    Evaluate final test
        ↓
    Save model, preprocessor, and metadata
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import pandas as pd

from src.preprocessing.preprocessor import (
    fit_preprocessor,
    prepare_features,
    transform_data,
)
from src.models.train import train_model, predict
from src.evaluation.metrics import evaluate_predictions


# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_FILE = PROJECT_ROOT / "data" / "processed" / "features" / "train.parquet"
VALID_FILE = PROJECT_ROOT / "data" / "processed" / "features" / "validation.parquet"
TEST_FILE = PROJECT_ROOT / "data" / "processed" / "features" / "test.parquet"

MODEL_DIR = PROJECT_ROOT / "artifacts" / "model"
PREPROCESSOR_DIR = PROJECT_ROOT / "artifacts" / "preprocessing"

MODEL_FILE = MODEL_DIR / "xgb_model.pkl"
PREPROCESSOR_FILE = PREPROCESSOR_DIR / "preprocessor.pkl"
METADATA_FILE = PROJECT_ROOT / "artifacts" / "model_metadata.json"


# ------------------------------------------------------------------
# Data loading
# ------------------------------------------------------------------

def load_datasets():
    """Load train, validation, and test feature datasets."""

    train_df = pd.read_parquet(TRAIN_FILE)
    valid_df = pd.read_parquet(VALID_FILE)
    test_df = pd.read_parquet(TEST_FILE)

    return train_df, valid_df, test_df


# ------------------------------------------------------------------
# Main training pipeline
# ------------------------------------------------------------------

def main():

    print("=" * 60)
    print("RETAIL DEMAND FORECASTING - MODEL TRAINING")
    print("=" * 60)

    # --------------------------------------------------------------
    # 1. Load datasets
    # --------------------------------------------------------------

    print("\n[1/6] Loading datasets...")

    train_df, valid_df, test_df = load_datasets()

    print(f"Train      : {train_df.shape}")
    print(f"Validation : {valid_df.shape}")
    print(f"Test       : {test_df.shape}")

    # --------------------------------------------------------------
    # 2. Separate features and target
    # --------------------------------------------------------------

    print("\n[2/6] Preparing features...")

    X_train, y_train = prepare_features(train_df)
    X_valid, y_valid = prepare_features(valid_df)
    X_test, y_test = prepare_features(test_df)

    print(f"Features: {X_train.shape[1]}")
    print(f"Target  : sales_quantity")

    # --------------------------------------------------------------
    # 3. Fit preprocessing ONLY on training data
    # --------------------------------------------------------------

    print("\n[3/6] Fitting preprocessing pipeline...")

    preprocessor, categorical_cols, numerical_cols = (
        fit_preprocessor(X_train)
    )

    X_train_processed = transform_data(
        preprocessor,
        X_train,
    )

    X_valid_processed = transform_data(
        preprocessor,
        X_valid,
    )

    X_test_processed = transform_data(
        preprocessor,
        X_test,
    )

    print(
        f"Processed train shape      : "
        f"{X_train_processed.shape}"
    )

    print(
        f"Processed validation shape : "
        f"{X_valid_processed.shape}"
    )

    print(
        f"Processed test shape       : "
        f"{X_test_processed.shape}"
    )

    # --------------------------------------------------------------
    # 4. Train selected XGBoost model
    # --------------------------------------------------------------

    print("\n[4/6] Training XGBoost model...")

    start_time = time.perf_counter()

    model = train_model(
        X_train_processed,
        y_train.to_numpy(),
    )

    training_time = time.perf_counter() - start_time

    print(
        f"Training completed in "
        f"{training_time:.2f} seconds"
    )

    # --------------------------------------------------------------
    # 5. Validation evaluation
    # --------------------------------------------------------------

    print("\n[5/6] Evaluating validation set...")

    valid_predictions = predict(
        model,
        X_valid_processed,
    )

    validation_metrics = evaluate_predictions(
        y_valid.to_numpy(),
        valid_predictions,
    )

    print("\nValidation metrics:")

    for metric, value in validation_metrics.items():
        print(f"{metric.upper():<6}: {value:.6f}")

    # --------------------------------------------------------------
    # 6. Final test evaluation + save artifacts
    # --------------------------------------------------------------

    print("\n[6/6] Final test evaluation...")

    test_predictions = predict(
        model,
        X_test_processed,
    )

    test_metrics = evaluate_predictions(
        y_test.to_numpy(),
        test_predictions,
    )

    print("\nTest metrics:")

    for metric, value in test_metrics.items():
        print(f"{metric.upper():<6}: {value:.6f}")

    # --------------------------------------------------------------
    # Save artifacts
    # --------------------------------------------------------------

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    PREPROCESSOR_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        model,
        MODEL_FILE,
    )

    joblib.dump(
        preprocessor,
        PREPROCESSOR_FILE,
    )

    metadata = {
        "model": "XGBRegressor",
        "parameters": {
            "n_estimators": 200,
            "max_depth": 6,
            "learning_rate": 0.10,
        },
        "target": "sales_quantity",
        "categorical_features": categorical_cols,
        "numerical_features": numerical_cols,
        "training_time_seconds": training_time,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "train_rows": len(train_df),
        "validation_rows": len(valid_df),
        "test_rows": len(test_df),
    }

    with open(
        METADATA_FILE,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            indent=4,
        )

    print("\nArtifacts saved:")

    print(f"Model        : {MODEL_FILE}")
    print(f"Preprocessor : {PREPROCESSOR_FILE}")
    print(f"Metadata     : {METADATA_FILE}")

    print("\nTraining pipeline completed successfully.")


if __name__ == "__main__":
    main()