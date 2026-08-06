from sqlalchemy import Boolean, Date, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from ..base import Base


class Calendar(Base):
    __tablename__ = "calendar"

    d: Mapped[str] = mapped_column(String(10), primary_key=True)

    date: Mapped[str] = mapped_column(Date)

    wm_yr_wk: Mapped[int] = mapped_column(Integer)

    weekday: Mapped[str] = mapped_column(String(20))

    wday: Mapped[int] = mapped_column(Integer)

    month: Mapped[int] = mapped_column(Integer)

    year: Mapped[int] = mapped_column(Integer)

    event_name_1: Mapped[str | None] = mapped_column(String(50), nullable=True)

    event_type_1: Mapped[str | None] = mapped_column(String(50), nullable=True)

    event_name_2: Mapped[str | None] = mapped_column(String(50), nullable=True)

    event_type_2: Mapped[str | None] = mapped_column(String(50), nullable=True)

    snap_CA: Mapped[bool] = mapped_column(Boolean)

    snap_TX: Mapped[bool] = mapped_column(Boolean)

    snap_WI: Mapped[bool] = mapped_column(Boolean)