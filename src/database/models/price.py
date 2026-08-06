from sqlalchemy import Float, ForeignKey, Integer, PrimaryKeyConstraint, String
from sqlalchemy.orm import Mapped, mapped_column

from ..base import Base


class Price(Base):
    __tablename__ = "prices"

    store_id: Mapped[str] = mapped_column(
        String(10),
        ForeignKey("stores.store_id")
    )

    item_id: Mapped[str] = mapped_column(
        String(50),
        ForeignKey("products.item_id")
    )

    wm_yr_wk: Mapped[int] = mapped_column(Integer)

    sell_price: Mapped[float] = mapped_column(Float)

    __table_args__ = (
        PrimaryKeyConstraint(
            "store_id",
            "item_id",
            "wm_yr_wk",
            name="pk_prices"
        ),
    )