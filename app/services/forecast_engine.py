"""
Aluminium Price Forecast Engine.

FORMULA REFERENCE
─────────────────
P_next = P_current
         + [(AMS_Q - AMS_Q-1) × PWt]
         + (PPI_Factor × P_current)

Where:
  AMS_Q     = (MC_Q  × DF_c) + CNG_Q
  AMS_Q-1   = (MC_Q-1 × DF_c) + CNG_Q-1

  MC_Q      = avg(LME_q1, LME_q2, LME_q3) + avg(MWP_q1, MWP_q2, MWP_q3)
              for the three months of the CURRENT quarter

  MC_Q-1    = same calculation for the PREVIOUS quarter

  PPI_Factor = (PPI_Q - PPI_Q-1) / PPI_Q-1
               where PPI_Q   = PPI of the LAST month of current quarter
                     PPI_Q-1 = PPI of the LAST month of previous quarter

  CNG_Q     = CNG of the LAST month of current quarter
  CNG_Q-1   = CNG of the LAST month of previous quarter

  DF_c      = 1.44 (constant)
  PWt       = part weight in lbs (per part number)

ITERATION LOGIC
───────────────
We are sitting in Feb 2026 (current month).
We predict March 2026 first, then April, …, up to Jan 2027 (12 months).
Each predicted price becomes the P_current for the next step.

The quarter context (MC_Q, PPI_Q, CNG_Q) is fixed per calendar quarter,
so months within the same quarter share the same quarterly inputs but each
uses the rolling predicted price as its P_current.
"""

import logging
from calendar import monthrange
from datetime import date, datetime, timezone
from typing import Optional

from app.core.config import settings
from app.data.base import MarketDataRepository, PartRepository
from app.models.response import (
    ForecastResponse,
    MonthForecast,
    QuarterContext,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# QUARTER HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _quarter_of_month(month: int) -> int:
    """Return 1-4 for a given month number (1-12)."""
    return (month - 1) // 3 + 1


def _quarter_months(year: int, quarter: int) -> list[str]:
    """
    Return the three 'YYYY-MM' keys that make up a calendar quarter.

    Quarter 1 → Jan, Feb, Mar  (months 1, 2, 3)
    Quarter 2 → Apr, May, Jun  (months 4, 5, 6)
    Quarter 3 → Jul, Aug, Sep  (months 7, 8, 9)
    Quarter 4 → Oct, Nov, Dec  (months 10, 11, 12)
    """
    first_month = (quarter - 1) * 3 + 1
    return [f"{year}-{m:02d}" for m in range(first_month, first_month + 3)]


def _last_month_of_quarter(year: int, quarter: int) -> str:
    """Return the 'YYYY-MM' key of the last month of a quarter."""
    last_month = quarter * 3
    return f"{year}-{last_month:02d}"


def _prev_quarter(year: int, quarter: int) -> tuple[int, int]:
    """Return (year, quarter) of the preceding quarter."""
    if quarter == 1:
        return year - 1, 4
    return year, quarter - 1


def _quarter_label(year: int, quarter: int) -> str:
    return f"Q{quarter}-{year}"


# ─────────────────────────────────────────────────────────────────────────────
# DATA FETCH HELPERS  (raise ValueError if data is unavailable)
# ─────────────────────────────────────────────────────────────────────────────

def _require(value: Optional[float], description: str) -> float:
    if value is None:
        raise ValueError(f"Required data unavailable: {description}")
    return value


def _compute_mc(
    year: int,
    quarter: int,
    market_repo: MarketDataRepository,
) -> float:
    """
    MC_Q = avg(LME over quarter months) + avg(Midwest over quarter months).
    """
    months = _quarter_months(year, quarter)
    lme_values = [
        _require(market_repo.get_lme(m), f"LME for {m}") for m in months
    ]
    mwp_values = [
        _require(market_repo.get_midwest_premium(m), f"Midwest premium for {m}")
        for m in months
    ]
    mc = sum(lme_values) / 3 + sum(mwp_values) / 3
    logger.debug(
        "MC_%s_%d: LME_avg=%.4f  MWP_avg=%.4f  MC=%.4f",
        f"Q{quarter}", year,
        sum(lme_values) / 3,
        sum(mwp_values) / 3,
        mc,
    )
    return mc


def _compute_quarter_context(
    year: int,
    quarter: int,
    market_repo: MarketDataRepository,
    df_c: float,
) -> QuarterContext:
    """
    Build the full QuarterContext (current + previous quarter) for a given
    target quarter.  Raises ValueError if any required data is missing.
    """
    prev_year, prev_quarter = _prev_quarter(year, quarter)

    # ── Current quarter ───────────────────────────────────────────────────
    mc_q = _compute_mc(year, quarter, market_repo)

    last_month_q = _last_month_of_quarter(year, quarter)
    ppi_q = _require(market_repo.get_ppi(last_month_q), f"PPI for {last_month_q}")
    cng_q = _require(market_repo.get_cng(last_month_q), f"CNG for {last_month_q}")
    ams_q = (mc_q * df_c) + cng_q

    # ── Previous quarter ─────────────────────────────────────────────────
    mc_q_prev = _compute_mc(prev_year, prev_quarter, market_repo)

    last_month_q_prev = _last_month_of_quarter(prev_year, prev_quarter)
    ppi_q_prev = _require(
        market_repo.get_ppi(last_month_q_prev), f"PPI for {last_month_q_prev}"
    )
    cng_q_prev = _require(
        market_repo.get_cng(last_month_q_prev), f"CNG for {last_month_q_prev}"
    )
    ams_q_prev = (mc_q_prev * df_c) + cng_q_prev

    # ── Derived factors ──────────────────────────────────────────────────
    if ppi_q_prev == 0:
        raise ValueError(f"PPI_Q-1 is zero for {last_month_q_prev} — cannot compute PPI_Factor")
    ppi_factor = (ppi_q - ppi_q_prev) / ppi_q_prev
    ams_delta = ams_q - ams_q_prev

    logger.debug(
        "QuarterContext %s: MC_Q=%.4f AMS_Q=%.4f | MC_Q-1=%.4f AMS_Q-1=%.4f | "
        "PPI_Factor=%.6f AMS_delta=%.4f",
        _quarter_label(year, quarter),
        mc_q, ams_q, mc_q_prev, ams_q_prev, ppi_factor, ams_delta,
    )

    return QuarterContext(
        quarter_label=_quarter_label(year, quarter),
        mc_q=round(mc_q, 6),
        ppi_q=round(ppi_q, 4),
        cng_q=round(cng_q, 6),
        ams_q=round(ams_q, 6),
        prev_quarter_label=_quarter_label(prev_year, prev_quarter),
        mc_q_prev=round(mc_q_prev, 6),
        ppi_q_prev=round(ppi_q_prev, 4),
        cng_q_prev=round(cng_q_prev, 6),
        ams_q_prev=round(ams_q_prev, 6),
        ppi_factor=round(ppi_factor, 8),
        ams_delta=round(ams_delta, 6),
    )


# ─────────────────────────────────────────────────────────────────────────────
# MONTH LABEL HELPER
# ─────────────────────────────────────────────────────────────────────────────

_MONTH_NAMES = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]

def _month_label(year_month: str) -> str:
    year, month = year_month.split("-")
    return f"{_MONTH_NAMES[int(month) - 1]} {year}"


def _advance_month(year: int, month: int) -> tuple[int, int]:
    """Return (year, month) one calendar month later."""
    if month == 12:
        return year + 1, 1
    return year, month + 1


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class ForecastEngine:
    """
    Stateless forecasting engine.  Depends only on the abstract repository
    interfaces — no concrete data source is referenced here.
    """

    def __init__(
        self,
        market_repo: MarketDataRepository,
        part_repo: PartRepository,
    ) -> None:
        self._market = market_repo
        self._parts = part_repo
        self._df_c = settings.DF_C
        self._horizon = settings.FORECAST_HORIZON_MONTHS

    # ─────────────────────────────────────────────────────────────────────
    # PUBLIC API
    # ─────────────────────────────────────────────────────────────────────

    def forecast(
        self,
        part_number: str,
        tier_1: str,
    ) -> ForecastResponse:
        """
        Produce a 12-month price forecast for *part_number* + *tier_1*.

        Parameters
        ----------
        part_number:
            The part to forecast.
        tier_1:
            Supplier name (e.g. "Kadon Aerospace", "NA").
            Together with part_number forms the unique lookup key.

        Returns
        -------
        ForecastResponse

        Raises
        ------
        ValueError
            If the part+tier_1 combination is unknown or price unavailable.
        """
        now_utc = datetime.now(timezone.utc)

        base_year_month = now_utc.strftime("%Y-%m")
        logger.info(
            "Starting forecast: part=%s  tier_1=%s  current_month=%s",
            part_number, tier_1, base_year_month,
        )

        # ── Part weight ───────────────────────────────────────────────────
        pwt = self._parts.get_part_weight(part_number, tier_1)
        if pwt is None:
            raise ValueError(
                f"Part not found: part_number='{part_number}' tier_1='{tier_1}'. "
                "Check that both values match the data store exactly."
            )

        # ── Base price — always from data store ───────────────────────────
        base_price = self._parts.get_base_price(part_number, tier_1, base_year_month)
        if base_price is None:
            raise ValueError(
                f"Current price not available for part_number='{part_number}' "
                f"tier_1='{tier_1}' month={base_year_month}."
            )

        logger.info(
            "Part=%s  Tier1=%s  PWt=%.2f lbs  P_base=%.4f $/lb",
            part_number, tier_1, pwt, base_price,
        )

        # ── Iterate over 12 future months ─────────────────────────────────
        base_year, base_month = map(int, base_year_month.split("-"))
        current_price = base_price
        forecasts: list[MonthForecast] = []

        # Cache quarter contexts so we don't recompute for every month in the
        # same quarter (LME/PPI/CNG lookup + avg is idempotent but costly).
        _quarter_cache: dict[tuple[int, int], QuarterContext] = {}

        forecast_year, forecast_month = _advance_month(base_year, base_month)

        for step in range(self._horizon):
            year_month_key = f"{forecast_year}-{forecast_month:02d}"
            quarter = _quarter_of_month(forecast_month)
            cache_key = (forecast_year, quarter)

            logger.debug("Step %d: forecasting %s", step + 1, year_month_key)

            # ── Build quarter context (cached per quarter) ────────────────
            if cache_key not in _quarter_cache:
                try:
                    ctx = _compute_quarter_context(
                        forecast_year, quarter, self._market, self._df_c
                    )
                    _quarter_cache[cache_key] = ctx
                except ValueError as exc:
                    raise ValueError(
                        f"Cannot compute quarter context for {year_month_key}: {exc}"
                    ) from exc
            else:
                ctx = _quarter_cache[cache_key]

            # ── Core formula ──────────────────────────────────────────────
            # P_next = P_current + [(AMS_Q - AMS_Q-1) × PWt] + (PPI_Factor × P_current)
            price_adjustment = ctx.ams_delta * pwt
            ppi_adjustment = ctx.ppi_factor * current_price
            predicted_price = current_price + price_adjustment + ppi_adjustment

            logger.debug(
                "%s: P_current=%.4f  AMS_delta=%.4f  PWt=%.2f  "
                "price_adj=%.4f  ppi_adj=%.4f  P_predicted=%.4f",
                year_month_key,
                current_price,
                ctx.ams_delta,
                pwt,
                price_adjustment,
                ppi_adjustment,
                predicted_price,
            )

            forecasts.append(
                MonthForecast(
                    year_month=year_month_key,
                    month_label=_month_label(year_month_key),
                    predicted_price=round(predicted_price, 4),
                    base_price_used=round(current_price, 4),
                    pwt=pwt,
                    df_c=self._df_c,
                    quarter_context=ctx,
                    is_data_projected=(step >= 1),  # Mar is first projection; beyond that, chained
                )
            )

            # Roll price forward for the next iteration
            current_price = predicted_price
            forecast_year, forecast_month = _advance_month(forecast_year, forecast_month)

        logger.info(
            "Forecast complete for part=%s: %d months, "
            "range %s → %s, price %.4f → %.4f",
            part_number,
            len(forecasts),
            forecasts[0].year_month,
            forecasts[-1].year_month,
            forecasts[0].predicted_price,
            forecasts[-1].predicted_price,
        )

        return ForecastResponse(
            part_number=part_number,
            tier_1=tier_1,
            pwt_lbs=pwt,
            base_year_month=base_year_month,
            base_price=base_price,
            forecast_generated_at=now_utc.isoformat(),
            forecasts=forecasts,
        )