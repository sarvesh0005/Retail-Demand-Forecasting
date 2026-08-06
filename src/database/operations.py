from sqlalchemy import text
import pandas as pd

from .connection import get_engine


def test_connection():
    """
    Test whether PostgreSQL is reachable.
    """

    engine = get_engine()

    with engine.connect() as conn:

        version = conn.execute(
            text("SELECT version();")
        )

        print(version.fetchone()[0])
        print("\nDatabase Connected Successfully!")


def execute_query(query: str):
    """
    Execute any SQL query.
    """

    engine = get_engine()

    with engine.connect() as conn:
        conn.execute(text(query))
        conn.commit()


def fetch_dataframe(query: str) -> pd.DataFrame:
    """
    Execute a SELECT query and return
    the result as a pandas DataFrame.
    """

    engine = get_engine()

    return pd.read_sql(query, engine)


def list_tables():

    query = """
    SELECT table_name
    FROM information_schema.tables
    WHERE table_schema='public'
    ORDER BY table_name;
    """

    tables = fetch_dataframe(query)

    return tables 