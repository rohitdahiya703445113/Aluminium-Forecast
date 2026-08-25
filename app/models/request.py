"""Pydantic request models for the forecast API."""

from pydantic import BaseModel, Field


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