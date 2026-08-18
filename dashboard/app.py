import requests
import pandas as pd
import streamlit as st


API_URL = "http://127.0.0.1:8000"


# ------------------------------------------------------------------
# Page configuration
# ------------------------------------------------------------------

st.set_page_config(
    page_title="Retail Demand Intelligence",
    page_icon="📊",
    layout="wide",
)


# ------------------------------------------------------------------
# API helpers
# ------------------------------------------------------------------

def get_summary():
    response = requests.get(
        f"{API_URL}/summary",
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def get_trend(
    store_id=None,
    cat_id=None,
    item_id=None,
):
    params = {}

    if store_id:
        params["store_id"] = store_id

    if cat_id:
        params["cat_id"] = cat_id

    if item_id:
        params["item_id"] = item_id

    response = requests.get(
        f"{API_URL}/demand/trend",
        params=params,
        timeout=60,
    )

    response.raise_for_status()

    return response.json()


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


# ------------------------------------------------------------------
# Header
# ------------------------------------------------------------------

st.title("📊 Retail Demand Intelligence")

st.markdown(
    """
Explore historical retail demand across products and stores,
identify high-demand entities, and understand demand patterns.
"""
)


# ------------------------------------------------------------------
# API connection
# ------------------------------------------------------------------

try:
    summary = get_summary()

except Exception as exc:

    st.error(
        "Unable to connect to the FastAPI backend."
    )

    st.code(
        str(exc)
    )

    st.stop()


# ------------------------------------------------------------------
# KPI cards
# ------------------------------------------------------------------

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


# ------------------------------------------------------------------
# Filters
# ------------------------------------------------------------------

st.subheader("Demand Analysis")

col1, col2, col3 = st.columns(3)

with col1:

    store_id = st.text_input(
        "Store ID",
        value="",
        placeholder="e.g. CA_1",
    )

with col2:

    cat_id = st.text_input(
        "Category",
        value="",
        placeholder="e.g. FOODS",
    )

with col3:

    item_id = st.text_input(
        "Product ID",
        value="",
        placeholder="e.g. FOODS_1_001",
    )


# ------------------------------------------------------------------
# Demand trend
# ------------------------------------------------------------------

try:

    trend_data = get_trend(
        store_id=store_id or None,
        cat_id=cat_id or None,
        item_id=item_id or None,
    )

    if trend_data:

        trend_df = pd.DataFrame(
            trend_data
        )

        trend_df["date"] = pd.to_datetime(
            trend_df["date"]
        )

        trend_df = trend_df.set_index(
            "date"
        )

        st.line_chart(
            trend_df["sales_quantity"]
        )

    else:

        st.info(
            "No demand data found for the selected filters."
        )

except Exception as exc:

    st.error(
        "Unable to load demand trend."
    )

    st.code(
        str(exc)
    )


st.divider()


# ------------------------------------------------------------------
# Rankings
# ------------------------------------------------------------------

st.subheader("Top Demand Rankings")

col1, col2, col3 = st.columns(3)


def display_ranking(
    column,
    title,
    ranking_type,
):

    with column:

        st.markdown(
            f"**{title}**"
        )

        try:

            data = get_rankings(
                ranking_type,
                top_n=10,
            )

            if data:

                ranking_df = pd.DataFrame(
                    data
                )

                ranking_df = ranking_df[
                    [
                        "rank",
                        "entity",
                        "total_sales",
                    ]
                ]

                ranking_df = ranking_df.rename(
                    columns={
                        "rank": "Rank",
                        "entity": "Entity",
                        "total_sales": "Sales",
                    }
                )

                st.dataframe(
                    ranking_df,
                    use_container_width=True,
                    hide_index=True,
                )

            else:

                st.info(
                    "No ranking data available."
                )

        except Exception as exc:

            st.error(
                "Unable to load ranking."
            )

            st.code(
                str(exc)
            )


display_ranking(
    col1,
    "Top Categories",
    "category",
)

display_ranking(
    col2,
    "Top Stores",
    "store",
)

display_ranking(
    col3,
    "Top Products",
    "product",
)


# ------------------------------------------------------------------
# Footer
# ------------------------------------------------------------------

st.divider()

st.caption(
    "Retail Demand Intelligence • "
    "XGBoost forecasting system"
)