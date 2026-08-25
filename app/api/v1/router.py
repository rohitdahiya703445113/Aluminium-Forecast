"""Aggregate all v1 endpoint routers. /forecast (hardcoded) removed."""

from fastapi import APIRouter

from app.api.v1.endpoints.forecast_excel import router as forecast_excel_router
from app.api.v1.endpoints.forecast_batch import router as forecast_batch_router

api_v1_router = APIRouter(prefix="/api/v1")
api_v1_router.include_router(forecast_excel_router)
api_v1_router.include_router(forecast_batch_router)