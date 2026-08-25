"""
Excel-backed market data + part master repository.

Reads aluminium_data.xlsx (5 sheets: Parts, LME, Midwest, PPI, CNG).

PARTS SHEET CONTRACT
────────────────────
Columns (by position):
  0 : Part Number
  1 : Tier 1          ← supplier name (e.g. "Kadon Aerospace", "NA")
  2 : Weight (lbs)
  3 : Current Price ($)

The unique key for every part lookup is (Part Number, Tier 1).

SWAP INSTRUCTIONS
─────────────────
Implement MarketDataRepository / PartRepository from app/data/base.py
and update DI in app/api/v1/endpoints/forecast_excel.py.
"""

import logging
import os
from functools import lru_cache
from typing import Optional

import pandas as pd

from app.data.base import MarketDataRepository, PartRepository

logger = logging.getLogger(__name__)

_DEFAULT_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "aluminium_data.xlsx")
EXCEL_FILE_PATH: str = os.environ.get("ALUMINIUM_EXCEL_PATH", _DEFAULT_PATH)


# ─────────────────────────────────────────────────────────────────────────────
# CACHED LOADER
# ─────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _load_workbook_data(path: str) -> dict[str, pd.DataFrame]:
    logger.info("Loading Excel data from: %s", path)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Excel data file not found at '{path}'. "
            "Set ALUMINIUM_EXCEL_PATH environment variable to the correct path."
        )
    sheets = pd.read_excel(
        path,
        sheet_name=["Parts", "LME", "Midwest", "PPI", "CNG"],
        engine="openpyxl",
        dtype=str,          # read everything as str first; we cast per column
        keep_default_na=False,  # prevent 'NA', 'N/A' etc. from becoming NaN
        na_values=[""],         # only treat blank cells as NaN
    )
    logger.info("Excel workbook loaded (%d sheets)", len(sheets))
    return sheets


def _invalidate_cache() -> None:
    _load_workbook_data.cache_clear()
    logger.info("Excel data cache cleared")


def _get_sheets() -> dict[str, pd.DataFrame]:
    return _load_workbook_data(EXCEL_FILE_PATH)


# ─────────────────────────────────────────────────────────────────────────────
# MARKET DATA HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _month_lookup(sheet_name: str) -> dict[str, float]:
    """Return {YYYY-MM: float} from a 2-column market sheet."""
    df = _get_sheets()[sheet_name]
    month_col = df.columns[0]
    value_col = df.columns[1]
    result: dict[str, float] = {}
    for _, row in df.iterrows():
        key = str(row[month_col]).strip()
        try:
            result[key] = float(row[value_col])
        except (ValueError, TypeError):
            logger.warning("Non-numeric value in %s for month %s", sheet_name, key)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# PARTS SHEET HELPER
# ─────────────────────────────────────────────────────────────────────────────

def _get_parts_df() -> pd.DataFrame:
    return _get_sheets()["Parts"]


def _make_part_key(part_number: str, tier_1: str) -> tuple[str, str]:
    return (part_number.strip(), tier_1.strip())


def _find_part_row(part_number: str, tier_1: str) -> Optional[pd.Series]:
    """
    Find the row in the Parts sheet matching (Part Number, Tier 1).
    Returns None if not found.
    """
    df = _get_parts_df()
    pn_col = df.columns[0]   # Part Number
    t1_col = df.columns[1]   # Tier 1

    # Fill NaN in Tier 1 column with empty string before comparison
    pn_series = df[pn_col].fillna("").astype(str).str.strip()
    t1_series = df[t1_col].fillna("").astype(str).str.strip()

    mask = (pn_series == part_number.strip()) & (t1_series == tier_1.strip())
    match = df[mask]
    if match.empty:
        return None
    return match.iloc[0]


def get_all_parts() -> list[dict]:
    """
    Return all rows from the Parts sheet as a list of dicts.
    Used by the /parts listing endpoint.
    """
    df = _get_parts_df()
    pn_col    = df.columns[0]
    tier1_col = df.columns[1]
    wt_col    = df.columns[2]
    price_col = df.columns[3]

    parts = []
    for _, row in df.iterrows():
        try:
            import math
            wt_val    = row[wt_col]
            price_val = row[price_col]
            if str(wt_val) in ("nan", "NaN", "") or str(price_val) in ("nan", "NaN", ""):
                continue
            parts.append({
                "part_number":   str(row[pn_col]).strip(),
                "tier_1":        str(row[tier1_col]).strip() if str(row[tier1_col]) != "nan" else "NA",
                "weight_lbs":    float(wt_val),
                "current_price": float(price_val),
            })
        except (ValueError, TypeError):
            logger.warning(
                "Skipping malformed row in Parts sheet: %s", dict(row)
            )
    return parts


# ─────────────────────────────────────────────────────────────────────────────
# CONCRETE IMPLEMENTATIONS
# ─────────────────────────────────────────────────────────────────────────────

class ExcelMarketDataRepository(MarketDataRepository):
    """Reads LME, Midwest, PPI, CNG from aluminium_data.xlsx (all $/lb)."""

    def _lookup(self, sheet: str, year_month: str, label: str) -> Optional[float]:
        value = _month_lookup(sheet).get(year_month)
        if value is None:
            logger.warning("%s data missing in Excel for %s", label, year_month)
        return value

    def get_lme(self, year_month: str) -> Optional[float]:
        return self._lookup("LME", year_month, "LME")

    def get_midwest_premium(self, year_month: str) -> Optional[float]:
        return self._lookup("Midwest", year_month, "Midwest premium")

    def get_ppi(self, year_month: str) -> Optional[float]:
        return self._lookup("PPI", year_month, "PPI")

    def get_cng(self, year_month: str) -> Optional[float]:
        return self._lookup("CNG", year_month, "CNG")


class ExcelPartRepository(PartRepository):
    """
    Reads part master from the Parts sheet.
    Lookup key = (Part Number, Tier 1) — both must match.
    """

    _PRICE_MONTH = "2026-08"   # month for which prices are stored in Excel

    def get_part_weight(self, part_number: str, tier_1: str) -> Optional[float]:
        row = _find_part_row(part_number, tier_1)
        if row is None:
            logger.warning(
                "Part not found in Excel: part_number=%s tier_1=%s",
                part_number, tier_1,
            )
            return None
        try:
            return float(row.iloc[2])   # Weight column
        except (ValueError, TypeError):
            logger.warning("Invalid weight for %s / %s", part_number, tier_1)
            return None

    def get_base_price(self, part_number: str, tier_1: str, year_month: str) -> Optional[float]:
        if year_month != self._PRICE_MONTH:
            return None
        row = _find_part_row(part_number, tier_1)
        if row is None:
            return None
        try:
            val = row.iloc[3]           # Current Price column
            return float(val) if pd.notna(val) else None
        except (ValueError, TypeError):
            logger.warning("Invalid price for %s / %s", part_number, tier_1)
            return None