"""
Demand trend analytics sanity test.
"""

from datetime import date

from src.analytics.trends import (
    get_demand_trend,
)


def main():

    print("=" * 60)
    print("DEMAND TREND TEST")
    print("=" * 60)

    # Test a specific item-store series.
    trend = get_demand_trend(
        item_id="FOODS_1_001",
        store_id="CA_1",
    )

    print(
        f"\nRows returned: {len(trend)}"
    )

    print("\nSample:")
    for row in trend[:5]:
        print(row)

    assert len(trend) > 0

    assert "date" in trend[0]

    assert (
        "sales_quantity"
        in trend[0]
    )

    # Test date filtering.
    filtered = get_demand_trend(
        item_id="FOODS_1_001",
        store_id="CA_1",
        start_date=date(2013, 1, 1),
        end_date=date(2013, 1, 31),
    )

    print(
        f"\nFiltered rows: "
        f"{len(filtered)}"
    )

    assert len(filtered) > 0

    print(
        "\nDemand trend test passed."
    )


if __name__ == "__main__":
    main()