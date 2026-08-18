import requests
import pandas as pd
import streamlit as st


API_URL = "http://127.0.0.1:8000"

DATA_FILE = (
    "data/processed/training_dataset/train_dataset.parquet"
)


# ==================================================================
# PAGE CONFIGURATION
# ==================================================================

st.set_page_config(
    page_title="Retail Demand Intelligence",
    page_icon="📊",
    layout="wide",
)


# ==================================================================
# API HELPERS
# ==================================================================

@st.cache_data(ttl=300)
def get_summary():

    response = requests.get(
        f"{API_URL}/summary",
        timeout=30,
    )

    response.raise_for_status()

    return response.json()


@st.cache_data(ttl=300)
def get_trend(
    store_id=None,
    cat_id=None,
    item_id=None,
    start_date=None,
    end_date=None,
):

    params = {}

    if store_id:
        params["store_id"] = store_id

    if cat_id:
        params["cat_id"] = cat_id

    if item_id:
        params["item_id"] = item_id

    if start_date:
        params["start_date"] = str(start_date)

    if end_date:
        params["end_date"] = str(end_date)

    response = requests.get(
        f"{API_URL}/demand/trend",
        params=params,
        timeout=60,
    )

    response.raise_for_status()

    return response.json()


@st.cache_data(ttl=300)
def get_rankings(
    ranking_type,
    top_n=10,
):

    response = requests.get(
        f"{API_URL}/rankings",
        params={
            "ranking_type": ranking_type,
            "top_n": top_n,
        },
        timeout=60,
    )

    response.raise_for_status()

    return response.json()


@st.cache_data
def load_reference_data():

    df = pd.read_parquet(
        DATA_FILE,
        columns=[
            "item_id",
            "store_id",
            "cat_id",
            "date",
        ],
    )

    df["date"] = pd.to_datetime(
        df["date"]
    )

    return df


# ==================================================================
# HEADER
# ==================================================================

st.title("📊 Retail Demand Intelligence")

st.markdown(
    """
**Explore retail demand, compare stores and products,
and understand historical demand patterns.**
"""
)


# ==================================================================
# API CONNECTION
# ==================================================================

try:

    summary = get_summary()

except Exception as exc:

    st.error(
        "Unable to connect to the FastAPI backend."
    )

    st.code(str(exc))

    st.stop()


# ==================================================================
# KPI SECTION
# ==================================================================

st.subheader("Business Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total Sales",
    f"{summary['total_sales']:,.0f}",
)

col2.metric(
    "Average Daily Demand",
    f"{summary['average_daily_demand']:,.1f}",
)

col3.metric(
    "Top Category",
    summary["top_category"],
)

col4.metric(
    "Top Store",
    summary["top_store"],
)


st.divider()


# ==================================================================
# LOAD REFERENCE DATA
# ==================================================================

try:

    reference_df = load_reference_data()

except Exception as exc:

    st.error(
        "Unable to load reference data."
    )

    st.code(str(exc))

    st.stop()


# ==================================================================
# DEMAND EXPLORER
# ==================================================================

st.subheader("🔎 Demand Explorer")

st.caption(
    "Filter the historical demand data by store, category, "
    "and product."
)


# ------------------------------------------------------------------
# Filters
# ------------------------------------------------------------------

col1, col2, col3 = st.columns(3)


with col1:

    store_options = sorted(
        reference_df["store_id"]
        .dropna()
        .unique()
        .tolist()
    )

    selected_store = st.selectbox(
        "Store",
        options=["All"] + store_options,
    )


with col2:

    category_options = sorted(
        reference_df["cat_id"]
        .dropna()
        .unique()
        .tolist()
    )

    selected_category = st.selectbox(
        "Category",
        options=["All"] + category_options,
    )


with col3:

    # Restrict product choices based on category.
    product_df = reference_df.copy()

    if selected_category != "All":

        product_df = product_df[
            product_df["cat_id"]
            == selected_category
        ]

    product_options = sorted(
        product_df["item_id"]
        .dropna()
        .unique()
        .tolist()
    )

    selected_product = st.selectbox(
        "Product",
        options=["All"] + product_options,
    )


# ==================================================================
# DATE FILTER
# ==================================================================

min_date = reference_df["date"].min().date()
max_date = reference_df["date"].max().date()

selected_dates = st.date_input(
    "Date range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)


if isinstance(
    selected_dates,
    tuple,
) and len(selected_dates) == 2:

    start_date, end_date = selected_dates

else:

    start_date = min_date
    end_date = max_date


# ==================================================================
# HISTORICAL DEMAND TREND
# ==================================================================

st.markdown("### Historical Demand")

try:

    trend_data = get_trend(
        store_id=(
            None
            if selected_store == "All"
            else selected_store
        ),
        cat_id=(
            None
            if selected_category == "All"
            else selected_category
        ),
        item_id=(
            None
            if selected_product == "All"
            else selected_product
        ),
        start_date=start_date,
        end_date=end_date,
    )

    if trend_data:

        trend_df = pd.DataFrame(
            trend_data
        )

        trend_df["date"] = pd.to_datetime(
            trend_df["date"]
        )

        trend_df = trend_df.sort_values(
            "date"
        )

        trend_df = trend_df.set_index(
            "date"
        )

        st.line_chart(
            trend_df["sales_quantity"],
            height=400,
        )

        st.caption(
            f"{len(trend_df):,} daily observations"
        )

    else:

        st.info(
            "No demand data found for the selected filters."
        )

except Exception as exc:

    st.error(
        "Unable to load demand trend."
    )

    st.code(str(exc))


st.divider()


# ==================================================================
# RANKINGS
# ==================================================================

st.subheader("🏆 Demand Rankings")

ranking_col1, ranking_col2 = st.columns(2)


# ------------------------------------------------------------------
# Top Stores
# ------------------------------------------------------------------

with ranking_col1:

    st.markdown("### Top Stores")

    try:

        stores = get_rankings(
            ranking_type="store",
            top_n=10,
        )

        stores_df = pd.DataFrame(
            stores
        )

        stores_df = stores_df[
            [
                "rank",
                "entity",
                "total_sales",
            ]
        ]

        stores_df = stores_df.rename(
            columns={
                "rank": "Rank",
                "entity": "Store",
                "total_sales": "Total Sales",
            }
        )

        st.dataframe(
            stores_df,
            use_container_width=True,
            hide_index=True,
        )

    except Exception as exc:

        st.error(
            "Unable to load store rankings."
        )

        st.code(str(exc))


# ------------------------------------------------------------------
# Top Products
# ------------------------------------------------------------------

with ranking_col2:

    st.markdown("### Top Products")

    try:

        products = get_rankings(
            ranking_type="product",
            top_n=10,
        )

        products_df = pd.DataFrame(
            products
        )

        products_df = products_df[
            [
                "rank",
                "entity",
                "total_sales",
            ]
        ]

        products_df = products_df.rename(
            columns={
                "rank": "Rank",
                "entity": "Product",
                "total_sales": "Total Sales",
            }
        )

        st.dataframe(
            products_df,
            use_container_width=True,
            hide_index=True,
        )

    except Exception as exc:

        st.error(
            "Unable to load product rankings."
        )

        st.code(str(exc))


# ==================================================================
# CATEGORY RANKING
# ==================================================================

st.markdown("### Category Demand")

try:

    categories = get_rankings(
        ranking_type="category",
        top_n=10,
    )

    categories_df = pd.DataFrame(
        categories
    )

    if not categories_df.empty:

        categories_df = categories_df[
            [
                "rank",
                "entity",
                "total_sales",
            ]
        ]

        categories_df = categories_df.rename(
            columns={
                "rank": "Rank",
                "entity": "Category",
                "total_sales": "Total Sales",
            }
        )

        st.dataframe(
            categories_df,
            use_container_width=True,
            hide_index=True,
        )

except Exception as exc:

    st.error(
        "Unable to load category rankings."
    )

    st.code(str(exc))


st.divider()


# ==================================================================
# ML SECTION
# ==================================================================

st.subheader("🤖 Machine Learning")

st.info(
    """
The XGBoost inference service is available through the FastAPI
`/predict` endpoint.

The next dashboard module will provide a proper forecasting
workflow with historical context, forecast horizon, and predicted
demand. It will be implemented through the backend so that the
dashboard does not duplicate feature-engineering logic.
"""
)


# ==================================================================
# FOOTER
# ==================================================================

st.divider()

st.caption(
    "Retail Demand Intelligence • "
    "XGBoost • FastAPI • Streamlit"
)