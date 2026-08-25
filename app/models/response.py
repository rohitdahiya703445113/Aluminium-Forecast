"""
Pydantic response models.

Every field that feeds the formula is surfaced so the caller can
validate and debug individual intermediate values.
"""

from typing import Optional
from pydantic import BaseModel, Field


class QuarterContext(BaseModel):
    """Quarterly aggregates used in the price formula for one forecast step."""

    # Current quarter
    quarter_label: str = Field(
        ..., description="Human-readable label, e.g. 'Q1-2026'"
    )
    mc_q: float = Field(
        ..., description="MC_Q: avg(LME + Midwest) for current quarter ($/lb)"
    )
    ppi_q: float = Field(
        ..., description="PPI_Q: PPI of last month of current quarter"
    )
    cng_q: float = Field(
        ..., description="CNG_Q: CNG of last month of current quarter ($/lb)"
    )
    ams_q: float = Field(
        ..., description="AMS_Q = (MC_Q × DF_c) + CNG_Q ($/lb)"
    )

    # Previous quarter
    prev_quarter_label: str = Field(
        ..., description="Human-readable label, e.g. 'Q4-2025'"
    )
    mc_q_prev: float = Field(
        ..., description="MC_Q-1: avg(LME + Midwest) for previous quarter ($/lb)"
    )
    ppi_q_prev: float = Field(
        ..., description="PPI_Q-1: PPI of last month of previous quarter"
    )
    cng_q_prev: float = Field(
        ..., description="CNG_Q-1: CNG of last month of previous quarter ($/lb)"
    )
    ams_q_prev: float = Field(
        ..., description="AMS_Q-1 = (MC_Q-1 × DF_c) + CNG_Q-1 ($/lb)"
    )

    # Derived factors
    ppi_factor: float = Field(
        ..., description="PPI_Factor = (PPI_Q - PPI_Q-1) / PPI_Q-1"
    )
    ams_delta: float = Field(
        ..., description="AMS_Q - AMS_Q-1 ($/lb)"
    )


class MonthForecast(BaseModel):
    """Complete forecast result for a single month."""

    year_month: str = Field(
        ..., description="Month being forecast, format YYYY-MM"
    )
    month_label: str = Field(
        ..., description="Human-readable month label, e.g. 'March 2026'"
    )
    predicted_price: float = Field(
        ..., description="Forecasted aluminium price for this part ($/lb)"
    )
    base_price_used: float = Field(
        ...,
        description=(
            "P_current used as input to forecast this month. "
            "For the first forecast month this is the known actual price."
        ),
    )
    pwt: float = Field(..., description="Part weight in lbs (PWt)")
    df_c: float = Field(..., description="Density/conversion factor DF_c (constant)")
    quarter_context: QuarterContext = Field(
        ..., description="All intermediate quarterly variables used in the formula"
    )
    is_data_projected: bool = Field(
        ...,
        description=(
            "True when one or more input values for this month came from "
            "projected (not actual) data."
        ),
    )


class ForecastResponse(BaseModel):
    """Top-level API response for a 12-month price forecast."""

    part_number: str
    pwt_lbs: float = Field(..., description="Part weight in lbs")
    base_year_month: str = Field(
        ..., description="The month we are forecasting FROM (current known month)"
    )
    base_price: float = Field(
        ..., description="Known price at base_year_month ($/lb)"
    )
    forecast_generated_at: str = Field(
        ..., description="UTC timestamp when the forecast was computed"
    )
    forecasts: list[MonthForecast] = Field(
        ..., description="Month-by-month forecast for the next 12 months"
    )


class ErrorResponse(BaseModel):
    """Standard error envelope."""

    error: str
    detail: Optional[str] = None
