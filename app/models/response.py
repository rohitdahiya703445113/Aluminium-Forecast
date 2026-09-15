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
    lme_q_avg: float = Field(
        ..., description="avg(LME) for current quarter months ($/lb)"
    )
    midwest_q_avg: float = Field(
        ..., description="avg(Midwest premium) for current quarter months ($/lb)"
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
    lme_q_prev_avg: float = Field(
        ..., description="avg(LME) for previous quarter months ($/lb)"
    )
    midwest_q_prev_avg: float = Field(
        ..., description="avg(Midwest premium) for previous quarter months ($/lb)"
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

class PriceFactorBreakdown(BaseModel):
    """
    Contribution of a single factor to the price for one forecast month.
    dollar_effect        : how many dollars this factor added (or subtracted).
                           Positive = pushed price up, Negative = pushed price down.
    percentage_of_base   : dollar_effect expressed as a % of the base (current) price.
                           e.g. -0.56% means this factor exerted a 0.56% downward
                           pressure on the price. Always stable and interpretable —
                           unaffected by how large or small the net price change is.
    """
    dollar_effect:      float = Field(..., description="Dollar contribution of this factor ($)")
    percentage_of_base: float = Field(
        ...,
        description="Contribution as % of base price (dollar_effect / base_price × 100)"
    )

class PriceChangeBreakdown(BaseModel):
    """
    Full decomposition of what drove the price change for one forecast month.
    The four factors are mathematically exact and always sum to total_change:
      total_change = lme_effect + midwest_effect + cng_effect + ppi_effect
    Derivation:
      P_next - P_current
        = [(AMS_Q - AMS_Q-1) × PWt]  +  [PPI_Factor × P_current]
        = [(MC_Q - MC_Q-1) × DF_c + (CNG_Q - CNG_Q-1)] × PWt  +  PPI_effect
      MC_Q - MC_Q-1 = (lme_q_avg - lme_q_prev_avg) + (midwest_q_avg - midwest_q_prev_avg)
      So:
        lme_effect     = (lme_q_avg - lme_q_prev_avg) × DF_c × PWt
        midwest_effect = (midwest_q_avg - midwest_q_prev_avg) × DF_c × PWt
        cng_effect     = (CNG_Q - CNG_Q-1) × PWt
        ppi_effect     = PPI_Factor × P_current
    """
    base_price:      float = Field(..., description="P_current used as base for this month ($)")
    predicted_price: float = Field(..., description="Predicted price for this month ($)")
    total_change:    float = Field(..., description="predicted_price - base_price ($)")
    lme_effect:     PriceFactorBreakdown = Field(..., description="Contribution from LME price movement")
    midwest_effect: PriceFactorBreakdown = Field(..., description="Contribution from Midwest premium movement")
    cng_effect:     PriceFactorBreakdown = Field(..., description="Contribution from CNG cost movement")
    ppi_effect:     PriceFactorBreakdown = Field(..., description="Contribution from PPI inflation")
    sum_check:      float = Field(
        ...,
        description=(
            "lme + midwest + cng + ppi — should equal total_change exactly. "
            "Any non-zero value here indicates a rounding artefact."
        ),
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
    price_change_breakdown: Optional[PriceChangeBreakdown] = Field(
        default=None,
        description=(
            "Factor-level decomposition of what drove the price change. "
            "Present on the /forecast-excel single-part endpoint. "
            "Null on batch and ML endpoints."
        ),
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
    tier_1: str = Field(..., description="Tier 1 supplier name")
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