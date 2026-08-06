from urllib.parse import quote_plus

from sqlalchemy import create_engine

from .config import DB_CONFIG

# Encode special characters in password (e.g., @, :, /, #)
password = quote_plus(DB_CONFIG["password"])

connection_string = (
    f"postgresql+psycopg2://"
    f"{DB_CONFIG['user']}:"
    f"{password}@"
    f"{DB_CONFIG['host']}:"
    f"{DB_CONFIG['port']}/"
    f"{DB_CONFIG['database']}"
)

engine = create_engine(
    connection_string,
    echo=False
)