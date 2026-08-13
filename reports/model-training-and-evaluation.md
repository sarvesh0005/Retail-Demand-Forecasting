# Model Training & Evaluation

## 1. Training Setup

The model-training stage used the feature-engineered train, validation, and test datasets stored as Parquet files.

**Target:** `sales_quantity`  
**Forecasting unit:** `(item_id, store_id)`

### Preprocessing

- Numerical and binary features were passed through without scaling because the selected model is tree-based.
- Categorical features (`weekday`, `dept_id`, `cat_id`, `state_id`) were encoded using `OneHotEncoder`.
- `handle_unknown="ignore"` was used to handle categories not seen during training.
- `item_id` and `store_id` were retained as identifiers rather than one-hot encoded because of their higher cardinality.
- The `ColumnTransformer` was fitted **only on the training data** and then used to transform validation and test data, preventing preprocessing leakage.

## 2. Modeling Strategy

A simple weekly seasonal-naive baseline was established first:

`prediction = sales_lag_7`

This represents the assumption that today's demand is similar to demand seven days earlier.

The primary ML model was `XGBRegressor`. XGBoost was selected because the dataset is structured/tabular and contains nonlinear demand relationships represented through lag, rolling, calendar, price, and categorical features.

The baseline configuration was:

| Parameter | Value |
|---|---:|
| `n_estimators` | 200 |
| `max_depth` | 6 |
| `learning_rate` | 0.10 |

A small improvement experiment used:

| Parameter | Value |
|---|---:|
| `n_estimators` | 300 |
| `max_depth` | 6 |
| `learning_rate` | 0.08 |
| `subsample` | 0.8 |
| `colsample_bytree` | 0.8 |
| `min_child_weight` | 3 |

No exhaustive hyperparameter search was performed because the objective was to establish a practical MVP rather than optimize for a competition leaderboard.

## 3. Validation Results

| Model | MAE | RMSE | WAPE | Training Time (s) |
|---|---:|---:|---:|---:|
| Seasonal-naive (`sales_lag_7`) | 1.139903 | 2.340855 | — | 0.00 |
| **XGBoost baseline** | **0.899425** | **1.683803** | **0.756598** | **12.81** |
| XGBoost improved | 0.901473 | 1.686051 | 0.758321 | 19.39 |

### Interpretation

The baseline XGBoost model clearly outperformed the seasonal-naive baseline.

Validation MAE decreased from **1.1399 to 0.8994**, approximately a **21% reduction**. RMSE also decreased substantially from **2.3409 to 1.6838**.

This indicates that the engineered features provided predictive information beyond simply using demand from the previous week.

The additional XGBoost configuration did not improve validation performance:

- MAE: `0.8994 → 0.9015`
- RMSE: `1.6838 → 1.6861`
- WAPE: `0.7566 → 0.7583`

It also required more training time. Therefore, the simpler baseline configuration was retained.

## 4. Final Test Evaluation

After model selection using validation performance, the selected baseline XGBoost model was evaluated on the held-out test period.

**Test period:** `2014-09-13 → 2014-12-31`  
**Test observations:** `165,000`

| Model | MAE | RMSE |
|---|---:|---:|
| Seasonal-naive (`sales_lag_7`) | 1.1938 | 2.4972 |
| **XGBoost baseline** | **0.9357** | **1.7649** |

**Final XGBoost WAPE:** `0.8051`

### Interpretation

The XGBoost model again outperformed the seasonal-naive baseline on the unseen test period.

Test MAE improved from **1.1938 to 0.9357**, approximately a **21.6% reduction**. RMSE also improved from **2.4972 to 1.7649**.

The consistency between validation and test results is important: the model's advantage was not limited to the validation period. It also generalized to the later unseen time period.

This should be interpreted as a successful **MVP baseline**, not as a fully optimized forecasting system. Extensive hyperparameter tuning was intentionally deferred because the model already demonstrated clear value over the simple forecasting baseline.

## 5. Final Model Decision

**Selected model:** XGBoost baseline

```text
n_estimators = 200
max_depth = 6
learning_rate = 0.10
```

The model was selected because it:

1. Outperformed the seasonal-naive baseline.
2. Achieved the best validation performance among the tested XGBoost configurations.
3. Also outperformed the naive baseline on the held-out test period.
4. Required less training time than the more complex configuration.
5. Provides a strong but maintainable starting point for the MVP.

## 6. Saved Artifacts

The training workflow produced:

```text
artifacts/
├── model/
│   └── xgb_model.pkl
├── preprocessing/
│   └── preprocessor.pkl
└── model_metadata.json
```

The preprocessing and model artifacts were reloaded after saving to verify that they could be used for inference.

The intended inference flow is:

```text
New data
   ↓
Feature engineering
   ↓
Saved preprocessor
   ↓
Saved XGBoost model
   ↓
Demand prediction
```

## 7. Engineering Takeaway

The main result of this stage is not only the final error value. The experiment established a reproducible modeling workflow with:

- temporal train/validation/test separation,
- leakage-safe preprocessing,
- a meaningful seasonal baseline,
- XGBoost model comparison,
- validation-based model selection,
- held-out test evaluation,
- and persisted model/preprocessing artifacts.

The next step is to refactor the validated notebook logic into reusable Python modules for the production-oriented ML pipeline.
