from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column

from ..base import Base


class Product(Base):
    __tablename__ = "products"

    item_id: Mapped[str] = mapped_column(String(50), primary_key=True)

    dept_id: Mapped[str] = mapped_column(String(30))

    cat_id: Mapped[str] = mapped_column(String(30))