from sqlalchemy import ForeignKey, Integer, PrimaryKeyConstraint, String
from sqlalchemy.orm import Mapped, mapped_column

from ..base import Base


class Sale(Base):
    __tablename__ = "sales"

    item_id: Mapped[str] = mapped_column(
        String(50),
        ForeignKey("products.item_id")
    )

    store_id: Mapped[str] = mapped_column(
        String(10),
        ForeignKey("stores.store_id")
    )

    d: Mapped[str] = mapped_column(
        String(10),
        ForeignKey("calendar.d")
    )

    sales_quantity: Mapped[int] = mapped_column(Integer)

    __table_args__ = (
        PrimaryKeyConstraint(
            "item_id",
            "store_id",
            "d",
            name="pk_sales"
        ),
    )