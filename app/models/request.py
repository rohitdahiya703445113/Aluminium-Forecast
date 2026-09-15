"""Pydantic request models for the forecast API."""

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class ForecastRequest(BaseModel):
    """Request body for the single-part forecast endpoint."""

    part_number: str = Field(
        ...,
        description="Aluminium part number (e.g. '09-0052-003')",
        examples=["09-0052-003"],
    )
    tier_1: str = Field(
        ...,
        description=(
            "Tier 1 supplier name exactly as it appears in the data store "
            "(e.g. 'Kadon Aerospace', 'Point Precision Inc.', 'NA'). "
            "Together with part_number this forms the unique lookup key."
        ),
        examples=["Kadon Aerospace"],
    )
    include_current_month: Literal["YES", "NO"] = Field(
        "NO",
        description=(
            "'YES' → forecast starts at the current month and covers 13 months "
            "(current month + next 12). 'NO' (default) → forecast starts next "
            "month and covers 12 months. Case-insensitive."
        ),
        examples=["NO"],
    )

    @field_validator("include_current_month", mode="before")
    @classmethod
    def _normalise_include_current_month(cls, v):
        return v.strip().upper() if isinstance(v, str) else v