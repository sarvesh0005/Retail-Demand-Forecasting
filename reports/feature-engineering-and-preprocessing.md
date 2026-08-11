# Feature Engineering and Preprocessing — Technical Engineering Report

**Project:** Retail Demand Forecasting (M5 Walmart-style dataset)
**Document type:** Internal engineering decision record
**Scope:** Feature engineering (`05_feature_engineering.ipynb`) and the planned preprocessing
approach for the upcoming model-training notebook
**Status:** Feature engineering complete and validated; preprocessing and model training not
yet implemented

This document records the actual engineering decisions made while building the feature set
for this project — what was implemented, why, what alternatives were considered, what problems
came up, and what remains open. It is a project record, not a tutorial.

---

## 1. Why Feature Engineering Was Necessary

The raw analytical dataset (`train_dataset.parquet`) contains one row per
`(item_id, store_id, date)` with the target `sales_quantity` and static/contemporaneous
context — product attributes, store attributes, calendar attributes, and price. On its own,
a single row only describes *that day's* conditions. It contains no information about how
demand for that item/store has been trending, whether last week looked similar, or whether
the current price represents a recent change.

Predicting `sales_quantity` requires signals a raw row cannot supply by itself:

- **Recent demand** — what did this specific item/store sell yesterday, and in the last few
  days.
- **Weekly demand patterns** — retail demand is strongly cyclical at a 7-day period (weekday
  vs. weekend effects).
- **Longer-term demand behavior** — a ~4-week window as a coarser signal of the recent demand
  level, less sensitive to single-day noise than a 1- or 7-day lag.
- **Calendar effects** — day-of-week, day-of-month, and seasonal position.
- **Price behavior** — whether the price changed recently and how the current price compares
  to what buyers have recently seen.
- **Product/store characteristics** — category, department, and state context.
- **Event-related context** — whether the day coincides with a known event.

This distinguishes the problem from ordinary (non-temporal) tabular ML: **row order and time
carry information that must be explicitly engineered into features**, and — critically —
**feature construction must respect temporal causality**. A feature describing "recent
demand" is only valid if it is built from observations that occurred *before* the row's own
date. This constraint (formalized in Section 8) shaped essentially every implementation
decision documented below.

---

## 2. Time-Series Structure

Each forecasting series in this dataset is defined by the pair `(item_id, store_id)` — one
distinct daily time series per product per store. The dataset is not one time series; it is
approximately 1,500 of them (300 products × 5 stores) stacked into a single table.

**Implementation:** the DataFrame is sorted by `item_id`, `store_id`, `date` (in that order)
before any lag or rolling computation is performed.

**Reason this ordering is required:** every lag and rolling feature in this project is built
with `groupby(["item_id", "store_id"])` followed by `.shift()` / `.rolling()`. These operations
act on rows **in the order they appear within each group** — they do not sort by date
internally. If the data were not pre-sorted chronologically within each `(item_id, store_id)`
group, a computed "lag 7" would not correspond to "seven days earlier"; it would silently
reflect whatever row happened to occupy that relative position, with no error raised. Sorting
once, up front, is what makes every later `.shift()`/`.rolling()` call correct by construction
rather than correct by coincidence.

**Reason features are generated independently per series:** demand levels, price levels, and
seasonality differ by item and by store — a lag or rolling statistic computed across series
boundaries would mix unrelated products/stores together, producing a feature with no
meaningful interpretation. `groupby(["item_id", "store_id"])` before every `.shift()`/
`.rolling()` call is the mechanism that enforces this; it is applied consistently across all
lag and rolling features (Sections 6–7).

---

## 3. Calendar Features

**Implemented features:** `day_of_week`, `day_of_month`, `week_of_year`, `month`, `quarter`,
`year`, and `is_event`.

`day_of_week`, `day_of_month`, `week_of_year`, and `quarter` are derived directly from the
`date` column. `month` and `year` are carried over from the source dataset rather than
recomputed. `is_event` is a binary indicator: `1` if either `event_name_1` or `event_name_2`
is non-null for that date, `0` otherwise.

**Role in this dataset, specifically:** the source data already includes `weekday` (a name
string) and `wday` (Walmart's fiscal weekday code, week starting Saturday) — neither is the
standard Monday-start integer encoding most gradient-boosted tree models split on efficiently.
`day_of_week` fills that specific gap for this project's planned model (XGBoost). `is_event`
was added because the raw `event_name_1`/`event_name_2` columns are high-cardinality
categorical strings that would complicate the encoding step (Section 15) for comparatively
little expected benefit at MVP stage; collapsing "is there an event at all" into one binary
column captures the coarse event-proximity signal without introducing that complexity. This
set was deliberately kept small — a handful of well-understood calendar features rather than
an exhaustive calendar feature library, consistent with the project's MVP scope.

**Leakage consideration:** none. Calendar attributes for any date — past or future — are
deterministic and known in advance; they carry no dependency on the target.

---

## 4. SNAP Features

**Retained as-is:** `snap_CA`, `snap_TX`, `snap_WI` — binary flags already present in the
source data, indicating SNAP (food-assistance) purchase eligibility on a given date, per
state.

**Reason for retention in this project:** the analytical subset spans stores in three states
(`CA`, `TX`, `WI`), and each store's `state_id` determines which of the three SNAP columns is
actually relevant to it. Rather than resolving this down to a single "snap active for this
store's state" column during feature engineering, all three flags were kept as-is and left for
the model-training stage — a tree-based model can learn the state-conditional relevance of
each flag on its own, and doing the reduction manually here would be an assumption about model
behavior made earlier than necessary. This is a design choice, not an empirically validated
one: no comparison was run between "three raw flags" and "one state-resolved flag."

---

## 5. Price Features

| Feature | Represents | Calculation (conceptual) | Rationale | Leakage consideration |
|---|---|---|---|---|
| `sell_price` | Current price for this item/store/day | Carried directly from the source data | Baseline price signal | None — contemporaneous, known at prediction time |
| `price_lag_1` | Most recently known price before today | `groupby(item_id, store_id)["sell_price"].shift(1)` | Reference point for detecting a price change | None — strictly backward-looking |
| `price_change` | Absolute price movement vs. yesterday's known price | `sell_price - price_lag_1` | Flags whether a price change just occurred | None — depends only on `sell_price` and `price_lag_1`, both known at time `t` |
| `price_change_pct` | Scale-invariant price movement | `price_change / price_lag_1` | A fixed absolute change means something different for a low-priced vs. high-priced item; this normalizes it | Guarded against division by zero (`price_lag_1 == 0` produces `NaN`, not `inf`) — an implementation necessity discovered while validating the feature (see Section 13) |
| `relative_price` | This store's price relative to peer stores carrying the same item, same day | `sell_price / mean(sell_price)` across stores, grouped by `(date, item_id)` | Captures competitive/relative pricing position | Compares only same-date, cross-store prices — contemporaneous information, not future information |

**On the assumed usefulness of these features:** their inclusion is a design assumption based
on domain reasoning (price and price changes plausibly influence demand), not an outcome of
any measured feature-importance or ablation experiment — no such experiment has been run yet
in this project. This is stated explicitly here rather than implied.

---

## 6. Lag Features

**Implemented:** `sales_lag_1`, `sales_lag_7`, `sales_lag_28` — the target's own value 1, 7,
and 28 days earlier, computed via `groupby(["item_id", "store_id"])["sales_quantity"].shift(n)`.

| Feature | Meaning |
|---|---|
| `sales_lag_1` | Previous-day demand |
| `sales_lag_7` | Previous-week demand (same weekday, one week prior) |
| `sales_lag_28` | Approximately previous-four-week demand |

**Reason these three specific windows were chosen for the MVP:** 1 day captures the most
immediate recent demand signal; 7 days aligns with the dataset's strong weekly seasonality
(weekday/weekend effects); 28 days (four weeks) gives a longer-horizon reference point without
introducing month-length irregularities that a literal "1 calendar month" lag would carry.
This is a deliberately small, standard set for an MVP — not an exhaustively tuned lag
selection; no systematic search over alternative lag windows (e.g. 14, 21, 56 days) was
performed.

**Why lag features matter specifically for this problem:** demand forecasting is
fundamentally a problem of extrapolating from a series' own recent history, and M5-style
retail demand is known to be highly autocorrelated at both daily and weekly periodicities —
lag features are the most direct way to expose that autocorrelation to a model that otherwise
sees each row independently.

**Implementation detail documented explicitly:** all three lags were computed **independently
per `(item_id, store_id)` series** via `groupby(...).shift(n)`, not via a plain column-level
`.shift()`. This was a deliberate implementation choice, not an oversight — a plain shift on
the stacked, multi-series DataFrame would pull rows from an adjacent, unrelated series once it
crossed a series boundary, silently corrupting the feature for the first `n` rows of every
series after the first.

---

## 7. Rolling Features

**Implemented:** `rolling_mean_7`, `rolling_mean_28`, `rolling_std_7`, `rolling_std_28` —
rolling mean and standard deviation of historical sales, over 7- and 28-day windows,
per series.

**What each represents:** `rolling_mean_7`/`rolling_mean_28` summarize the recent average
demand level at two different horizons; `rolling_std_7`/`rolling_std_28` summarize recent
demand *volatility* at the same two horizons — a series with a high rolling standard deviation
relative to its mean is behaving more erratically than one with a low ratio, which is
potentially useful signal independent of the mean level itself.

### Critical implementation decision: shift before rolling

The implementation explicitly applies `shift(1)` **before** `.rolling(window)`, rather than
applying a rolling statistic directly to `sales_quantity`:

```
shifted = groupby(item_id, store_id)["sales_quantity"].shift(1)
rolling_mean_7 = groupby(item_id, store_id)[shifted].rolling(7).mean()
```

**Why the shift is necessary:** a default rolling window ending at row `t` **includes** row
`t` itself. Since `sales_quantity` at row `t` is the prediction target, computing
`rolling_mean_7` directly on `sales_quantity` without first shifting would mean the target
value at time `t` is included in its own feature — a direct, severe form of target leakage.
Concretely: `rolling_mean_7(t)` must be built from `sales_quantity` at `t-1` through `t-7`,
and must **not** include `sales_quantity(t)`. Shifting by 1 first, then rolling, is what
enforces this: after the shift, the value aligned to row `t` is actually `sales_quantity` from
`t-1`, so a 7-window roll ending at row `t` covers exactly `t-1` through `t-7`.

This is documented here as the single most leakage-prone implementation step in the notebook,
and it is the step most extensively validated (Section 13).

---

## 8. Temporal Leakage Prevention

This project treats temporal leakage as the primary correctness risk in the feature-engineering
stage — more consequential than any individual feature's predictive value. The following
concrete safeguards were implemented:

| Safeguard | Implementation | What it prevents |
|---|---|---|
| Chronological sorting | `sort_values(["item_id", "store_id", "date"])` before any lag/rolling computation | Lag/rolling operations silently referencing rows out of date order |
| Group-wise lag creation | `groupby(["item_id", "store_id"]).shift(n)` for every lag feature | One series' history leaking into another series' lag feature |
| Shift before rolling | `shift(1)` applied before every `.rolling(window)` call | The current-row target value being included in its own rolling-window feature |
| No random train/test split | `train_test_split(..., shuffle=True)` was explicitly **not** used | A model being evaluated on data chronologically interleaved with its own training window, which would overstate real-world performance |
| Chronological train/validation/test split | Split computed from sorted unique `date` values, not row position | Ensures every training row's date genuinely precedes every validation/test row's date |
| Target exclusion from feature columns | Explicit assertion that `sales_quantity` is not present in the feature column list | The target being accidentally included as a model input |
| Preprocessing fit only on training data (planned) | `ColumnTransformer` to be `.fit()` on `X_train` only, `.transform()`-only on validation/test (Section 15) | Statistics derived from validation/test data (e.g. one-hot categories, if any were data-dependent) influencing the training process |

These are the concrete, implemented decisions behind "no leakage" in this project — not a
general leakage checklist. Each one addresses a specific, plausible failure mode identified
while building this exact feature set (e.g. the rolling-window inclusion risk in Section 7
was the direct motivation for the shift-before-roll pattern, not a generic best practice
applied without cause).

---

## 9. Zero Sales

Zero-sales observations (`sales_quantity == 0`) were intentionally retained throughout the
dataset — no filtering, downsampling, or exclusion based on the target value was applied at
any stage of feature engineering.

**Reason:** M5-style retail demand is intermittent by nature — most item/store/day
combinations legitimately have zero units sold, not due to a data defect. Removing zero-demand
rows would distort the demand distribution the model is trained against, understating how
often a series has no sales and biasing any resulting model toward over-predicting demand.
Preserving zero observations is a requirement for this problem, not an optional data-quality
step.

---

## 10. Missing Values

Lag and rolling features introduce structural missing values at the start of every
`(item_id, store_id)` series:

- `sales_lag_1`: missing for the first row of each series.
- `sales_lag_7`: missing for the first 7 rows of each series.
- `sales_lag_28` / `rolling_mean_28` / `rolling_std_28`: missing for the first 28 rows of each
  series.

**Cause:** these are structural, not anomalous — a series' first `n` rows have fewer than `n`
prior observations available, so the corresponding lag/rolling value cannot be computed. This
was confirmed by checking that missing `sales_lag_28` values are concentrated exactly in
row-within-series positions 0–27 for every series, not scattered elsewhere in the dataset.

**MVP decision:** these values were left as genuine `NaN` — **no imputation was implemented**.
Filling with `0` was explicitly rejected as a default: a `NaN` means "history not yet
available," which is a different condition from "history available and confirmed zero," and
conflating the two would introduce a systematic bias into exactly the rows where the model has
the least information to work with. The stated reason for deferring this to the model-training
stage is that XGBoost (the planned model, Section 15) handles `NaN` natively; no imputation
strategy has been evaluated or implemented in this project.

A related, project-specific detail: the subset's date range (Section 12) means this
start-of-series missingness affects the **first 28 days of the entire subset**, in addition to
any product/store combination that begins selling partway through the window — both are
expected consequences of a bounded date range, not implementation defects.

---

## 11. Feature Selection — Final Feature Groups

| Group | Columns |
|---|---|
| Calendar | `day_of_week`, `day_of_month`, `week_of_year`, `month`, `quarter`, `year` |
| Event | `is_event` |
| SNAP | `snap_CA`, `snap_TX`, `snap_WI` |
| Price | `sell_price`, `price_lag_1`, `price_change`, `price_change_pct`, `relative_price` |
| Historical demand | `sales_lag_1`, `sales_lag_7`, `sales_lag_28`, `rolling_mean_7`, `rolling_mean_28`, `rolling_std_7`, `rolling_std_28` |
| Categorical | `weekday`, `dept_id`, `cat_id`, `state_id` |

**Explicitly excluded from the final feature list:** the raw `event_name_1`, `event_type_1`,
`event_name_2`, `event_type_2` columns. These are categorical string fields with meaningful
cardinality; including them directly would have required deciding an encoding strategy for
them during feature engineering. `is_event` (Section 3) was judged sufficient signal for the
MVP, and any richer treatment of event type/name is deferred — if pursued later, it belongs in
preprocessing or model training, not in this notebook.

`item_id`, `store_id`, `d`, `date`, and the target `sales_quantity` are retained in the saved
Parquet files for traceability and per-series analysis, but are explicitly excluded from the
feature column list used to build `X_train`/`X_valid`/`X_test`.

---

## 12. Train / Validation / Test Split

`train_test_split(..., shuffle=True)` was explicitly not used, since a random shuffle would
allow training rows and validation/test rows to be temporally interleaved — inconsistent with
a forecasting evaluation, where a model must be assessed on its ability to predict genuinely
future, unseen dates.

**Implementation:** the sorted list of unique dates in the dataset was used to compute two
cutoff points at approximately the 70th and 85th percentile of unique dates, producing:

- **Train** — earliest date through the 70% cutoff.
- **Validation** — the following ~15% of dates.
- **Test** — the final ~15% of dates.

**Temporal requirement, enforced programmatically:** `max(train_date) < min(validation_date)`
and `max(validation_date) < min(test_date)` — both checked with explicit assertions in the
notebook, not just visually inspected.

**Purpose of each split:**
- **Train** — used to fit the model (and, later, the preprocessing pipeline — Section 15).
- **Validation** — used for model selection and hyperparameter comparison during the
  upcoming model-training phase.
- **Test** — held out entirely until final evaluation, to give an unbiased estimate of
  performance on genuinely future data.

**Actual split boundaries and row counts:** these depend on the specific run of
`05_feature_engineering.ipynb` against the current `train_dataset.parquet` and are recorded in
the notebook's own output and in the accompanying `metadata.json` file
(`data/processed/features/metadata.json`), which stores `train_start`/`train_end`,
`validation_start`/`validation_end`, `test_start`/`test_end`, and row counts per split. This
report does not restate those exact figures inline, to avoid the report and the generated
metadata drifting out of sync — `metadata.json` is the source of truth for split boundaries.

---

## 13. Validation and Sanity Checks

| Check | Purpose |
|---|---|
| Duplicate `(item_id, store_id, date)` check | Confirms the dataset's grain is exactly one row per series per day — a prerequisite for every `groupby`-based feature |
| Date-range validation | Compares the actual date range present in the loaded Parquet file against the requested extraction range (Section 12/project context), so a mismatch between what was requested and what was actually extracted is surfaced explicitly rather than silently assumed away |
| Lag correctness validation | Manually recomputes `sales_lag_7`/`sales_lag_28` for sampled series using plain positional indexing, independent of the `groupby().shift()` implementation, and asserts the two match row-for-row |
| Rolling-feature correctness validation | Manually recomputes `rolling_mean_7` for a sampled series using an explicit Python loop that excludes the current row from its own window, and asserts it matches the `shift(1)`-then-`rolling()` implementation |
| Missing-value inspection | Quantifies `NaN` counts per lag/rolling column and confirms they are concentrated at the start of each series (row-within-series position 0 through `window-1`), not scattered — evidence the missingness is structural, not a bug |
| Infinite-value inspection | Scans all numeric feature columns for `inf`/`-inf`, specifically because `price_change_pct`'s division could produce infinite values if not explicitly guarded |
| Target/feature separation | Explicit assertions that `sales_quantity` and the identifier columns are absent from the feature column list before building `X_train`/`X_valid`/`X_test` |
| Chronological split validation | Assertions that `max(train_date) < min(validation_date)` and `max(validation_date) < min(test_date)`, plus a row-count identity check that train + validation + test rows exactly equal the pre-split row count |
| Parquet round-trip validation | Reloads the saved `train.parquet` and confirms shape and dtypes match what was written, before considering the output usable by a downstream notebook |

Each check exists because it targets a specific, plausible failure mode of this
implementation (e.g. the infinite-value check exists *because* `price_change_pct` involves a
division), not as a generic data-science checklist applied without cause.

---

## 14. Parquet Outputs

The final feature dataset is saved as three files plus one metadata file, under
`data/processed/features/`:

- `train.parquet`
- `validation.parquet`
- `test.parquet`
- `metadata.json` — records the feature column list, target column, identifier columns, split
  date boundaries, and per-split row counts.

**Reason Parquet was chosen for this project:** it is a columnar, typed storage format —
column dtypes (including the `NaN`-carrying float columns produced by lag/rolling features)
round-trip exactly, which a CSV-based format would not guarantee (CSV requires re-inferring
types on every load). It also integrates directly with the `pandas`/`pyarrow` stack already in
use throughout this project's ETL pipeline, with no new tooling introduced. No specific
compression ratio or load-time performance improvement has been measured for this project —
the choice is based on Parquet's structural properties (typed, columnar), not on a benchmarked
comparison against CSV for this dataset.

---

## 15. Preprocessing Decision

**Feature engineering** (this notebook) constructs the forecasting-meaningful variables
described in Sections 3–7 — calendar, event, price, and historical-demand signals derived from
the raw analytical dataset. **Preprocessing** is a distinct, not-yet-implemented step that will
prepare those variables for a specific model — encoding categoricals, and (for models that
need it) scaling numerics. This project treats the two as separate pipeline stages
deliberately, so that the feature set itself remains model-agnostic and the preprocessing
choices can be revisited independently of how the features were derived.

**Current categorical features requiring encoding:** `weekday`, `dept_id`, `cat_id`,
`state_id`.

**Planned MVP preprocessing structure** (not yet implemented — this is the intended design for
the upcoming model-training notebook):

```
X_train
    │
    ├── numerical features   ──┐
    └── categorical features ──┤── ColumnTransformer
                                │        │
                                │        ▼
                                │   OneHotEncoder (categorical branch)
                                │
                                ▼
                           XGBoost Regressor
```

**Why preprocessing must be fit only on training data:** any preprocessing step whose behavior
depends on the data it sees (e.g. which categorical values a `OneHotEncoder` learns to expect)
must be fit exclusively on `X_train`, then applied to `X_validation`/`X_test` via
`.transform()` only — never re-fit on validation or test data. This is a direct extension of
the leakage-prevention principle already applied to the temporal split (Section 8): validation
and test data must remain information the model (and its preprocessing) has never influenced
in any way, not only never used for gradient updates.

Intended flow:

```
X_train      → fit_transform(preprocessor) → model training input
X_validation → transform(preprocessor)      → model selection input
X_test       → transform(preprocessor)      → final evaluation input
```

**Explicit status:** this preprocessing design is a plan for the next notebook, not a
completed implementation. No `ColumnTransformer`, encoder, or preprocessing pipeline has been
built or tested in this project as of this report.

---

## 16. Why Scaling Is Not Used

Standard scaling (`StandardScaler`) and min-max scaling (`MinMaxScaler`) were deliberately
excluded from the planned preprocessing approach for this project's numerical features.

**Reason:** the first planned model is XGBoost, a tree-based gradient boosting model. Tree
splits are based on feature value thresholds, not distances or gradients over feature
magnitude, so tree-based models are invariant to monotonic transformations like standardization
— scaling numerical inputs would add a preprocessing step with no expected benefit for this
specific model choice.

**Scope of this decision:** this is stated as an MVP engineering simplification specific to
the XGBoost baseline, not a general claim that scaling is unnecessary. If a distance-based or
gradient-descent-based model (e.g. linear regression, a neural network) were introduced later,
scaling would need to be reconsidered for that model specifically.

---

## 17. Notebook vs. Production Code

This project follows a consistent workflow for turning experimental notebook logic into
reusable code:

```
Notebook → Experiment → Validate → Confirm design → Refactor → Reusable .py module
```

This pattern has already been applied once, for the ETL stage: `extract.ipynb`,
`transform.ipynb`, and `load.ipynb` were used to discover and validate the extraction,
transformation, and loading logic, which was subsequently refactored into
`src/etl/extract.py`, `src/etl/transform.py`, and `src/etl/load.py`.

The same principle is intended for the ML stage. `05_feature_engineering.ipynb` is currently
the experimental implementation of feature construction — validated (Section 13) but not yet
refactored. After the modeling approach is validated in the upcoming model-training notebook,
the feature-engineering and preprocessing logic is intended to be extracted into reusable
modules, with a potential future structure of:

```
src/
├── etl/
│   ├── extract.py
│   ├── transform.py
│   └── load.py
│
├── features/
│   └── engineering.py
│
├── preprocessing/
│   └── preprocessor.py
│
├── models/
│   └── train.py
│
└── evaluation/
    └── metrics.py
```

**Explicit status:** only the `src/etl/` modules currently exist. `src/features/`,
`src/preprocessing/`, `src/models/`, and `src/evaluation/` are planned, not implemented — this
structure is documented here as intent, not as completed work.

---

## 18. Engineering Trade-offs

| # | Decision | Reason | Trade-off | Current status |
|---|---|---|---|---|
| 1 | ~1.1M-row structured MVP subset instead of the full ~58M-row dataset | Full-scale in-memory processing during ML experimentation is impractical and unnecessary for validating the approach | The MVP subset (5 stores, 300 products, ~2 years) may not capture every pattern present in the full dataset; results may not generalize identically at full scale | Adopted for all ML experimentation to date |
| 2 | PostgreSQL-side joins instead of pandas-side joins | A pandas-based join across the full dataset caused a `MemoryError` during the price join | Requires SQL query construction and database-side validation instead of purely in-Python logic | Adopted; joins for the analytical subset are performed in PostgreSQL |
| 3 | Structured, contiguous temporal subset instead of random sampling | Preserves continuous time-series structure, which lag/rolling features and chronological splitting require | A structured subset (fixed stores/products/date range) may not be as representative as a stratified random sample would be for other purposes | Adopted |
| 4 | Feature engineering implemented in a notebook rather than immediately modularized | Matches the project's notebook-first validation workflow (Section 17); avoids building reusable code around a design that might still change | Feature logic is not yet reusable/importable outside the notebook | Current state; modularization planned after modeling is validated |
| 5 | Simple lag/rolling features (1/7/28-day) instead of a more elaborate forecasting feature system | Sufficient signal for an MVP baseline; keeps the feature set small and fully validated | May omit useful signal (e.g. additional lag windows, exponential weighting, cross-series features) that a more complete system would include | Adopted for the MVP; no more advanced feature system has been evaluated |
| 6 | Simple categorical encoding (planned: one-hot) instead of target/mean encoding or embeddings | Straightforward, leakage-safe if fit only on training data (Section 15); appropriate for the current small categorical cardinality | Target/mean encoding could capture more nuanced categorical signal but introduces its own leakage risks and complexity | Planned, not yet implemented |
| 7 | XGBoost baseline instead of a more sophisticated forecasting model (e.g. deep learning, hierarchical forecasting) | Establishes a practical, well-understood baseline appropriate for an MVP and resume-oriented project | May not reach the performance ceiling of more sophisticated approaches | Planned as the first model; not yet implemented |

---

## 19. What We Deliberately Did Not Do

The following were intentionally excluded from this stage of the project. None are being
characterized as poor techniques in general — they were excluded because they are not
necessary for this project's current objective (a practical, understandable MVP):

- Loading or processing the full ~58M-row analytical dataset in pandas.
- Random sampling of the dataset (would break the continuous time-series structure required
  by lag/rolling features and chronological splitting).
- Spark, Dask, or Polars — the project remains on `pandas` + `pyarrow`/Parquet throughout.
- A complex feature-store system or elaborate feature-engineering framework/class hierarchy.
- Deep learning forecasting models.
- Advanced hyperparameter optimization.
- Target/mean encoding or embedding-based categorical encoding.
- Numerical feature scaling (not applicable to the planned tree-based model — Section 16).
- Kaggle-competition-specific optimization (e.g. extensive feature ensembles, model stacking,
  leaderboard-driven feature search).

---

## 20. Current State

**DONE:**
- PostgreSQL database with a normalized schema (`calendar`, `products`, `stores`, `prices`,
  `sales`).
- ETL pipeline (Extract, Transform, Load), validated and refactored into `src/etl/`.
- Controlled ~1.1M-row analytical subset (`data/processed/training_dataset/train_dataset.parquet`).
- Forecasting feature engineering (`05_feature_engineering.ipynb`): calendar, event, SNAP,
  price, and historical-demand (lag/rolling) features.
- Temporal leakage safeguards, implemented and validated (Sections 8, 13).
- Chronological train/validation/test split, with programmatically enforced non-overlap.
- Parquet outputs (`train.parquet`, `validation.parquet`, `test.parquet`, `metadata.json`)
  under `data/processed/features/`.

**NEXT:**
- Model-training notebook.
- Preprocessing pipeline (`ColumnTransformer` with `OneHotEncoder` for categoricals, fit on
  training data only).
- XGBoost baseline model.
- Validation metrics and evaluation on the held-out validation split.
- Model comparison / tuning, if pursued.
- Final evaluation on the held-out test split.
- Model artifact persistence.
- Refactoring the validated feature-engineering and preprocessing logic into
  `src/features/`, `src/preprocessing/`, `src/models/`, and `src/evaluation/`.
