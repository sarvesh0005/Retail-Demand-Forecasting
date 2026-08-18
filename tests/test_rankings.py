"""
Ranking analytics sanity test.
"""

from src.analytics.rankings import get_rankings


def validate_ranking(
    results,
    expected_max_count,
):
    """Validate a ranking result."""

    # Dataset may contain fewer entities than requested.
    assert 1 <= len(results) <= expected_max_count

    # Required fields.
    required_fields = {
        "rank",
        "entity",
        "total_sales",
    }

    for row in results:
        assert required_fields.issubset(
            row.keys()
        )

        assert row["rank"] >= 1
        assert row["total_sales"] >= 0

    # Rank should start at 1 and be sequential.
    ranks = [
        row["rank"]
        for row in results
    ]

    assert ranks == list(
        range(1, len(results) + 1)
    )

    # Results should be sorted descending
    # by total sales.
    sales = [
        row["total_sales"]
        for row in results
    ]

    assert sales == sorted(
        sales,
        reverse=True,
    )


def main():

    print("=" * 60)
    print("RANKINGS TEST")
    print("=" * 60)

    # --------------------------------------------------------------
    # Category rankings
    # --------------------------------------------------------------

    categories = get_rankings(
        ranking_type="category",
        top_n=5,
    )

    print("\nTop categories:")

    for row in categories:
        print(row)

    validate_ranking(
        categories,
        expected_max_count=5,
    )

    # --------------------------------------------------------------
    # Store rankings
    # --------------------------------------------------------------

    stores = get_rankings(
        ranking_type="store",
        top_n=5,
    )

    print("\nTop stores:")

    for row in stores:
        print(row)

    validate_ranking(
        stores,
        expected_max_count=5,
    )

    # --------------------------------------------------------------
    # Product rankings
    # --------------------------------------------------------------

    products = get_rankings(
        ranking_type="product",
        top_n=5,
    )

    print("\nTop products:")

    for row in products:
        print(row)

    validate_ranking(
        products,
        expected_max_count=5,
    )

    # --------------------------------------------------------------
    # Invalid top_n
    # --------------------------------------------------------------

    try:
        get_rankings(
            ranking_type="category",
            top_n=0,
        )

        raise AssertionError(
            "Expected ValueError for top_n=0."
        )

    except ValueError:
        pass

    # --------------------------------------------------------------
    # Invalid ranking type
    # --------------------------------------------------------------

    try:
        get_rankings(
            ranking_type="invalid",
            top_n=5,
        )

        raise AssertionError(
            "Expected ValueError for invalid ranking type."
        )

    except ValueError:
        pass

    print(
        "\nRankings test passed."
    )


if __name__ == "__main__":
    main()