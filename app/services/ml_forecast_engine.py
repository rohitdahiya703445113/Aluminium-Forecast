"""
ML-based Price Forecast Engine.

WHAT THIS DOES
──────────────
Identical iteration logic to ForecastEngine (forecast_engine.py) — same
quarter context computation, same chaining, same response shape — but the
predicted price for each month comes from a trained ML model (joblib pickle)
instead of the analytical formula.

MODEL INPUT COLUMNS (must match training exactly)
──────────────────────────────────────────────────
  Weight          → PWt in lbs (from part master)
  Current Price   → P_current (actual for step 1, chained prediction after)
  MC_Q            → LME + Midwest of the month being predicted ($/lb)
  MC_Q-1          → LME + Midwest at last month of previous quarter ($/lb)
  PPI_Q           → PPI index of the month being predicted
  PPI_Q-1         → PPI index at last month of previous quarter
  CNG_Q           → CNG cost of the month being predicted ($/lb)
  CNG_Q-1         → CNG cost at last month of previous quarter ($/lb)
  Drauss Factor   → DF_c constant (1.44)

ISOLATION GUARANTEE
────────────────────
This file imports ONLY from:
  - app.core.config          (settings / DF_c)
  - app.data.base            (abstract repository interfaces)
  - app.models.response      (shared response models — read-only)
  - Standard library + pandas + joblib

It does NOT import from forecast_engine.py or batch_forecast.py.
Deleting this file and its endpoint leaves everything else untouched.
"""

import logging
import os
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

from app.core.config import settings
from app.data.base import MarketDataRepository, PartRepository
from app.models.response import ForecastResponse, MonthForecast, QuarterContext

logger = logging.getLogger(__name__)

# ── Default model path — override via ML_MODEL_PATH env var ──────────────────
_DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "best_price_model.pkl"
)
ML_MODEL_PATH: str = os.environ.get("ML_MODEL_PATH", _DEFAULT_MODEL_PATH)

# ── Exact column names the model was trained on ───────────────────────────────
_MODEL_COLUMNS = [
    "Weight",
    "Current Price",
    "MC_Q",
    "MC_Q-1",
    "PPI_Q",
    "PPI_Q-1",
    "CNG_Q",
    "CNG_Q-1",
    "Drauss Factor",
]


# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADER  (lazy — loaded on first use, cached in memory)
# ─────────────────────────────────────────────────────────────────────────────

_cached_model = None


def _load_model():
    """
    Load the joblib model from disk, caching it after the first load.
    Raises FileNotFoundError with a clear message if the file is missing.
    """
    global _cached_model
    if _cached_model is not None:
        return _cached_model

    path = os.path.abspath(ML_MODEL_PATH)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"ML model file not found at '{path}'. "
            "Place 'best_price_model.pkl' next to main.py, or set the "
            "ML_MODEL_PATH environment variable to the correct path."
        )

    import joblib
    logger.info("Loading ML model from: %s", path)
    _cached_model = joblib.load(path)
    logger.info("ML model loaded successfully: %s", type(_cached_model).__name__)
    return _cached_model


def invalidate_model_cache() -> None:
    """Clear cached model so next request reloads from disk."""
    global _cached_model
    _cached_model = None
    logger.info("ML model cache cleared")


# ─────────────────────────────────────────────────────────────────────────────
# QUARTER HELPERS  (duplicated from forecast_engine.py intentionally —
#                  keeps this module fully self-contained)
# ─────────────────────────────────────────────────────────────────────────────

def _quarter_of_month(month: int) -> int:
    return (month - 1) // 3 + 1


def _quarter_months(year: int, quarter: int) -> list[str]:
    first = (quarter - 1) * 3 + 1
    return [f"{year}-{m:02d}" for m in range(first, first + 3)]


def _last_month_of_quarter(year: int, quarter: int) -> str:
    return f"{year}-{quarter * 3:02d}"


def _prev_quarter(year: int, quarter: int) -> tuple[int, int]:
    return (year - 1, 4) if quarter == 1 else (year, quarter - 1)


def _quarter_label(year: int, quarter: int) -> str:
    return f"Q{quarter}-{year}"


def _advance_month(year: int, month: int) -> tuple[int, int]:
    return (year + 1, 1) if month == 12 else (year, month + 1)


def _prev_month(year: int, month: int) -> tuple[int, int]:
    return (year - 1, 12) if month == 1 else (year, month - 1)


def _price_month_error(
    parts: PartRepository,
    part_number: str,
    tier_1: str,
    needed_month: str,
    include_current_month: bool,
) -> ValueError:
    """Explain why no P_current was found for *needed_month*."""
    data_month = parts.get_price_month()
    if data_month is not None and data_month != needed_month:
        flag = "YES" if include_current_month else "NO"
        return ValueError(
            f"Part prices in the data store are for {_month_label(data_month)}, but "
            f"include_current_month={flag} needs {_month_label(needed_month)} prices "
            "as P_current. Update the price column (and the month in its header) "
            "or change include_current_month."
        )
    return ValueError(
        f"Current price not available for part_number='{part_number}' "
        f"tier_1='{tier_1}' month={needed_month}. Make sure the price column "
        "header names its month, e.g. 'Current Price ($) Aug 2026'."
    )


_MONTH_NAMES = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]


def _month_label(year_month: str) -> str:
    year, month = year_month.split("-")
    return f"{_MONTH_NAMES[int(month) - 1]} {year}"


def _require(value: Optional[float], description: str) -> float:
    if value is None:
        raise ValueError(f"Required data unavailable: {description}")
    return value


# ─────────────────────────────────────────────────────────────────────────────
# QUARTER CONTEXT BUILDER  (same logic as forecast_engine._compute_quarter_context)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_quarter_context(
    year: int,
    month: int,
    market: MarketDataRepository,
    df_c: float,
) -> QuarterContext:
    quarter = _quarter_of_month(month)
    prev_year, prev_quarter = _prev_quarter(year, quarter)

    # ── Current (predicted) month ─────────────────────────────────────────
    target = f"{year}-{month:02d}"
    lme_q  = _require(market.get_lme(target),             f"LME for {target}")
    mwp_q  = _require(market.get_midwest_premium(target), f"Midwest for {target}")
    mc_q   = lme_q + mwp_q
    ppi_q  = _require(market.get_ppi(target), f"PPI for {target}")
    cng_q  = _require(market.get_cng(target), f"CNG for {target}")
    ams_q  = mc_q * df_c + cng_q

    # ── Previous quarter (last month) ─────────────────────────────────────
    last_qp   = _last_month_of_quarter(prev_year, prev_quarter)
    lme_qp    = _require(market.get_lme(last_qp),             f"LME for {last_qp}")
    mwp_qp    = _require(market.get_midwest_premium(last_qp), f"Midwest for {last_qp}")
    mc_q_prev = lme_qp + mwp_qp
    ppi_q_prev = _require(market.get_ppi(last_qp), f"PPI for {last_qp}")
    cng_q_prev = _require(market.get_cng(last_qp), f"CNG for {last_qp}")
    ams_q_prev = mc_q_prev * df_c + cng_q_prev

    if ppi_q_prev == 0:
        raise ValueError(f"PPI_Q-1 is zero for {last_qp}")
    ppi_factor = (ppi_q - ppi_q_prev) / ppi_q_prev
    ams_delta  = ams_q - ams_q_prev

    return QuarterContext(
        quarter_label=_quarter_label(year, quarter),
        mc_q=round(mc_q, 6),
        lme_q=round(lme_q, 6),
        midwest_q=round(mwp_q, 6),
        ppi_q=round(ppi_q, 4),
        cng_q=round(cng_q, 6),
        ams_q=round(ams_q, 6),
        prev_quarter_label=_quarter_label(prev_year, prev_quarter),
        mc_q_1=round(mc_q_prev, 6),
        lme_q_1=round(lme_qp, 6),
        midwest_q_1=round(mwp_qp, 6),
        ppi_q_1=round(ppi_q_prev, 4),
        cng_q_1=round(cng_q_prev, 6),
        ams_q_1=round(ams_q_prev, 6),
        ppi_factor=round(ppi_factor, 8),
        ams_delta=round(ams_delta, 6),
    )


# ─────────────────────────────────────────────────────────────────────────────
# ML FORECAST ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class MLForecastEngine:
    """
    Stateless ML-based forecast engine.

    Computes all intermediate quarterly variables (MC_Q, AMS_Q, PPI_Factor,
    CNG_Q, etc.) using the same logic as ForecastEngine, but uses the
    trained ML model to produce the predicted price instead of the formula.

    Injected dependencies (same interface as ForecastEngine):
      market_repo : MarketDataRepository
      part_repo   : PartRepository
    """

    def __init__(
        self,
        market_repo: MarketDataRepository,
        part_repo: PartRepository,
    ) -> None:
        self._market  = market_repo
        self._parts   = part_repo
        self._df_c    = settings.DF_C
        self._horizon = settings.FORECAST_HORIZON_MONTHS

    def forecast(
        self,
        part_number: str,
        tier_1: str,
        include_current_month: bool = False,
    ) -> ForecastResponse:
        """
        Produce a 12-month ML price forecast for part_number + tier_1.
        With include_current_month=True the forecast starts at the current
        month and covers 13 months (current month + next 12).
        P_current for the first step is this month's price (False) or last
        month's price (True); the data store must hold prices for that month.

        All quarterly inputs (MC_Q, PPI_Q, CNG_Q, etc.) are computed
        identically to the formula-based engine. The predicted price for
        each month is produced by the ML model using those inputs plus the
        chained current price and part weight.

        Raises
        ------
        FileNotFoundError
            If best_price_model.pkl is not found.
        ValueError
            If the part+tier_1 combination is unknown, price is unavailable,
            or any required market data is missing.
        """
        model = _load_model()   # raises FileNotFoundError if missing

        now_utc       = datetime.now(timezone.utc)
        current_month = now_utc.strftime("%Y-%m")
        # P_current = price of the month just before the first forecast month:
        #   include_current_month → LAST month's price, forecast starts this month
        #   otherwise             → THIS month's price, forecast starts next month
        if include_current_month:
            price_year, price_month = _prev_month(now_utc.year, now_utc.month)
            base_year_month = f"{price_year}-{price_month:02d}"
        else:
            base_year_month = current_month

        logger.info(
            "ML forecast: part=%s  tier_1=%s  current_month=%s  price_month=%s",
            part_number, tier_1, current_month, base_year_month,
        )

        # ── Part master ───────────────────────────────────────────────────
        pwt = self._parts.get_part_weight(part_number, tier_1)
        if pwt is None:
            raise ValueError(
                f"Part not found: part_number='{part_number}' tier_1='{tier_1}'. "
                "Check that both values match the data store exactly."
            )

        base_price = self._parts.get_base_price(part_number, tier_1, base_year_month)
        if base_price is None:
            raise _price_month_error(
                self._parts, part_number, tier_1, base_year_month, include_current_month
            )

        logger.info(
            "Part=%s  Tier1=%s  PWt=%.2f lbs  P_base=%.4f",
            part_number, tier_1, pwt, base_price,
        )

        # ── Iterate 12 months ─────────────────────────────────────────────
        base_year, base_month = map(int, base_year_month.split("-"))
        current_price = base_price
        forecasts: list[MonthForecast] = []

        # First forecast month is always the month after the price month.
        # include_current_month → 13 months (this month + next 12), else 12.
        forecast_year, forecast_month = _advance_month(base_year, base_month)
        n_months = self._horizon + 1 if include_current_month else self._horizon

        for step in range(n_months):
            year_month_key = f"{forecast_year}-{forecast_month:02d}"

            logger.debug("ML step %d: forecasting %s", step + 1, year_month_key)

            # ── Context for this month (MC_Q / PPI_Q vary per month) ──────
            try:
                ctx = _compute_quarter_context(
                    forecast_year, forecast_month, self._market, self._df_c
                )
            except ValueError as exc:
                raise ValueError(
                    f"Cannot compute quarter context for {year_month_key}: {exc}"
                ) from exc

            # ── Build model input row ─────────────────────────────────────
            input_df = pd.DataFrame([{
                "Weight":         pwt,
                "Current Price":  current_price,
                "MC_Q":           ctx.mc_q,
                "MC_Q-1":         ctx.mc_q_1,
                "PPI_Q":          ctx.ppi_q,
                "PPI_Q-1":        ctx.ppi_q_1,
                "CNG_Q":          ctx.cng_q,
                "CNG_Q-1":        ctx.cng_q_1,
                "Drauss Factor":  self._df_c,
            }], columns=_MODEL_COLUMNS)

            # ── Model prediction ──────────────────────────────────────────
            try:
                predicted_price = float(model.predict(input_df)[0])
            except Exception as exc:
                raise ValueError(
                    f"Model prediction failed for {year_month_key}: {exc}"
                ) from exc

            logger.debug(
                "%s: P_current=%.4f → ML_predicted=%.4f",
                year_month_key, current_price, predicted_price,
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
                    is_data_projected=(step >= 1),
                )
            )

            # Chain predicted price forward
            current_price = predicted_price
            forecast_year, forecast_month = _advance_month(forecast_year, forecast_month)

        logger.info(
            "ML forecast complete: part=%s  %s → %s  price %.4f → %.4f",
            part_number,
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