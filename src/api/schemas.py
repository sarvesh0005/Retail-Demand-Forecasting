from datetime import date

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """
    Request for demand prediction.

    The API receives the historical analytical context
    required to construct lag and rolling features.
    """

    data: list[dict] = Field(
        ...,
        min_length=29,
        description=(
            "Historical analytical observations for the "
            "item/store series, including the target sales_quantity."
        ),
    )


class PredictionResponse(BaseModel):
    item_id: str
    store_id: str
    date: date
    predicted_sales_quantity: float