"""
Ranking analytics for the Retail Demand Intelligence dashboard.
"""

from typing import Literal

from src.analytics.summary import load_analytical_data


RankingType = Literal[
    "category",
    "store",
    "product",
]


def get_rankings(
    ranking_type: RankingType,
    top_n: int = 10,
) -> list[dict]:
    """
    Return top entities ranked by total sales demand.

    Parameters
    ----------
    ranking_type:
        One of: category, store, product.

    top_n:
        Number of entities to return.
    """

    if top_n < 1 or top_n > 100:
        raise ValueError(
            "top_n must be between 1 and 100."
        )

    df = load_analytical_data()

    column_map = {
        "category": "cat_id",
        "store": "store_id",
        "product": "item_id",
    }

    if ranking_type not in column_map:
        raise ValueError(
            "ranking_type must be "
            "'category', 'store', or 'product'."
        )

    group_column = column_map[
        ranking_type
    ]

    rankings = (
        df.groupby(group_column)["sales_quantity"]
        .sum()
        .sort_values(ascending=False)
        .head(top_n)
        .reset_index()
    )

    rankings = rankings.rename(
        columns={
            group_column: "entity",
            "sales_quantity": "total_sales",
        }
    )

    rankings["total_sales"] = (
        rankings["total_sales"]
        .astype(float)
    )

    rankings["rank"] = range(
        1,
        len(rankings) + 1,
    )

    return rankings[
        [
            "rank",
            "entity",
            "total_sales",
        ]
    ].to_dict(
        orient="records"
    )