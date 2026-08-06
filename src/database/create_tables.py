from src.database.base import Base
from src.database.connection import engine

# Import all ORM models
import src.database.schema


def create_tables():
    Base.metadata.create_all(bind=engine)
    print("All database tables created successfully!")


if __name__ == "__main__":
    create_tables()