"""
ML-based forecast endpoint  –  /api/v1/forecast-ml

Identical request/response contract to /api/v1/forecast-excel.
The only difference: predicted_price comes from best_price_model.pkl
instead of the analytical formula.

This endpoint is intentionally isolated:
  - Its own router, own DI wiring, own engine class
  - Removing this file + unregistering from router.py leaves
    everything else completely untouched

Routes
──────
POST /api/v1/forecast-ml
    Single-part ML forecast. Body: { part_number, tier_1 }

POST /api/v1/forecast-ml/reload-model
    Clears the in-memory model cache so an updated pickle is picked up
    without restarting the server.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException, status

from app.data.base import MarketDataRepository, PartRepository
from app.data.excel_store import ExcelMarketDataRepository, ExcelPartRepository
from app.models.request import ForecastRequest
from app.models.response import ErrorResponse, ForecastResponse
from app.services.ml_forecast_engine import MLForecastEngine, invalidate_model_cache

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/forecast-ml", tags=["forecast-ml"])


# ─────────────────────────────────────────────────────────────────────────────
# DEPENDENCY INJECTION
# ─────────────────────────────────────────────────────────────────────────────

def get_market_repo() -> MarketDataRepository:
    return ExcelMarketDataRepository()


def get_part_repo() -> PartRepository:
    return ExcelPartRepository()


def get_ml_engine(
    market_repo: MarketDataRepository = Depends(get_market_repo),
    part_repo: PartRepository         = Depends(get_part_repo),
) -> MLForecastEngine:
    return MLForecastEngine(market_repo=market_repo, part_repo=part_repo)


# ─────────────────────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────────────────────

@router.post(
    "/reload-model",
    summary="Reload ML model from disk",
)
def reload_model() -> dict:
    """
    Clear the in-memory model cache.
    The next forecast request will reload `best_price_model.pkl` from disk.
    Useful after replacing the pickle file without restarting the server.
    """
    invalidate_model_cache()
    return {
        "status": "cache cleared",
        "message": "ML model will be reloaded from disk on the next request.",
    }


@router.post(
    "",
    response_model=ForecastResponse,
    summary="ML-based price forecast for one part",
    responses={
        200: {"description": "12-month ML forecast with all intermediate variables"},
        404: {"model": ErrorResponse, "description": "Part + Tier 1 not found"},
        422: {"model": ErrorResponse, "description": "Price unavailable or invalid input"},
        500: {"model": ErrorResponse, "description": "Model file missing or prediction error"},
    },
)
def get_forecast_ml(
    payload: ForecastRequest,
    engine: MLForecastEngine = Depends(get_ml_engine),
) -> ForecastResponse:
    """
    Forecast aluminium price for **part_number + tier_1** for the next 12 months
    using the trained ML model (`best_price_model.pkl`).

    **Same as `/api/v1/forecast-excel` except:**
    - `predicted_price` is produced by the ML model, not the analytical formula
    - Prices are chained: Sep prediction → Oct `Current Price` input → and so on
    - All intermediate variables (MC_Q, AMS_Q, PPI_Factor, etc.) are still
      computed from `aluminium_data.xlsx` using the same quarterly logic

    **Model input columns (in order):**
    `Weight | Current Price | MC_Q | MC_Q-1 | PPI_Q | PPI_Q-1 | CNG_Q | CNG_Q-1 | Drauss Factor`

    **Setup:** Place `best_price_model.pkl` next to `main.py`, or set the
    `ML_MODEL_PATH` environment variable to its full path.
    """
    logger.info(
        "ML forecast request: part_number=%s  tier_1=%s",
        payload.part_number, payload.tier_1,
    )

    try:
        result = engine.forecast(
            part_number=payload.part_number,
            tier_1=payload.tier_1,
        )

    except FileNotFoundError as exc:
        # Model pickle not found — actionable 500 with clear message
        logger.error("ML model file not found: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        )

    except ValueError as exc:
        error_msg = str(exc)
        logger.warning("ML forecast rejected: %s", error_msg)
        if "Part not found" in error_msg:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=error_msg,
            )
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=error_msg,
        )

    except Exception:
        logger.exception(
            "Unexpected ML forecast error: part=%s tier_1=%s",
            payload.part_number, payload.tier_1,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred during ML forecasting. See server logs.",
        )

    logger.info(
        "ML forecast served: part=%s tier_1=%s  price_range=%.4f–%.4f",
        payload.part_number, payload.tier_1,
        result.forecasts[0].predicted_price,
        result.forecasts[-1].predicted_price,
    )
    return result