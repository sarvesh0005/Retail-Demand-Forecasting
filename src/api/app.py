from datetime import date

import pandas as pd
from fastapi import FastAPI, HTTPException

from src.analytics.rankings import get_rankings
from src.analytics.summary import get_summary
from src.analytics.trends import get_demand_trend
from src.api.schemas import (
    PredictionRequest,
    PredictionResponse,
)
from src.inference.predict import predict_demand


app = FastAPI(
    title="Retail Demand Forecasting API",
    description=(
        "Production-oriented API for retail demand prediction "
        "and demand analytics."
    ),
    version="1.0.0",
)


# ------------------------------------------------------------------
# Root
# ------------------------------------------------------------------

@app.get("/")
def root():
    return {
        "message": "Retail Demand Forecasting API",
        "status": "running",
    }


# ------------------------------------------------------------------
# Health check
# ------------------------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "healthy",
    }


# ------------------------------------------------------------------
# Dashboard summary
# ------------------------------------------------------------------

@app.get("/summary")
def summary():
    return get_summary()


# ------------------------------------------------------------------
# Historical demand trend
# ------------------------------------------------------------------

@app.get("/demand/trend")
def demand_trend(
    store_id: str | None = None,
    cat_id: str | None = None,
    item_id: str | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
):
    return get_demand_trend(
        store_id=store_id,
        cat_id=cat_id,
        item_id=item_id,
        start_date=start_date,
        end_date=end_date,
    )


# ------------------------------------------------------------------
# Demand rankings
# ------------------------------------------------------------------

@app.get("/rankings")
def rankings(
    ranking_type: str,
    top_n: int = 10,
):
    try:
        return get_rankings(
            ranking_type=ranking_type,
            top_n=top_n,
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc


# ------------------------------------------------------------------
# Demand prediction
# ------------------------------------------------------------------

@app.post(
    "/predict",
    response_model=list[PredictionResponse],
)
def predict(request: PredictionRequest):

    try:
        analytical_df = pd.DataFrame(
            request.data
        )

        predictions = predict_demand(
            analytical_df
        )

        return predictions.to_dict(
            orient="records"
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc