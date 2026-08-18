"""
Analytics module sanity test.
"""

from src.analytics.summary import get_summary


def main():

    print("=" * 60)
    print("ANALYTICS SUMMARY TEST")
    print("=" * 60)

    summary = get_summary()

    print("\nSummary:")
    for key, value in summary.items():
        print(f"{key}: {value}")

    assert "total_sales" in summary
    assert "average_daily_demand" in summary
    assert "top_category" in summary
    assert "top_store" in summary

    assert summary["total_sales"] >= 0
    assert summary["average_daily_demand"] >= 0

    print(
        "\nAnalytics summary test passed."
    )


if __name__ == "__main__":
    main()