"""
CycloneX - Pydantic Request & Response Data Schemas
Defines structured input/output validation models for FastAPI endpoints.

DISCLAIMER: AI-assisted research prototype for decision support. Not an official IMD warning.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator


# --- System & Health Schemas ---
class HealthResponse(BaseModel):
    status: str = Field(..., json_schema_extra={"example": "healthy"})
    model_version: str = Field(..., json_schema_extra={"example": "cyclonex-v1.0.0"})
    active_models: List[str] = Field(..., json_schema_extra={"example": ["EfficientNetB0", "GRU_Track", "Multimodal_Fusion"]})
    disclaimer: str = Field(
        default="AI-assisted research prototype. Not an official IMD warning.",
        description="Mandatory advisory disclaimer"
    )


class SystemInfoResponse(BaseModel):
    system_name: str = "CycloneX Decision-Support API"
    problem_statement: str = "SIH 26070 (Ministry of Earth Sciences / IMD)"
    model_version: str = "cyclonex-v1.0.0"
    endpoints: List[str]
    intensity_classes: List[str]
    channels_supported: List[str]
    disclaimer: str = "AI-assisted research prototype. Not an official IMD warning."


# --- Intensity & Image Prediction Schemas ---
class SatelliteFrameRequest(BaseModel):
    cyclone_id: Optional[str] = Field(default="UNKNOWN_STORM", description="Storm identifier")
    timestamp: Optional[str] = Field(default=None, description="Observation timestamp UTC")
    image_base64: Optional[str] = Field(default=None, description="Base64 encoded 201x201 PNG/JPEG or NPY image")
    raw_array: Optional[List[List[List[float]]]] = Field(default=None, description="Nested 3D array (201, 201, 3)")
    return_gradcam: bool = Field(default=True, description="Whether to compute and return Grad-CAM overlay")
    gradcam_alpha: float = Field(default=0.45, ge=0.0, le=1.0, description="Grad-CAM blending transparency")


class IntensityPredictionResponse(BaseModel):
    cyclone_id: str
    predicted_vmax_kt: float = Field(..., description="Estimated maximum sustained wind speed in Knots")
    intensity_category: str = Field(..., description="Weak | Moderate | Strong | Very Strong")
    confidence_interval: Dict[str, float] = Field(..., description="Lower and upper bound in knots")
    gradcam_overlay_base64: Optional[str] = Field(default=None, description="Base64 PNG data URI of Grad-CAM overlay on IR1")
    model_version: str
    disclaimer: str = "AI-assisted decision-support research prototype. Not an official IMD warning."


# --- Track Forecasting Schemas ---
class TrackObservation(BaseModel):
    timestamp: Optional[str] = None
    lat: float = Field(..., ge=-90.0, le=90.0, description="Latitude in degrees")
    lon: float = Field(..., ge=-180.0, le=180.0, description="Longitude in degrees")
    vmax: float = Field(..., ge=0.0, le=250.0, description="Wind speed in knots")
    pres: Optional[float] = Field(default=1000.0, description="Central pressure in hPa")


class TrackPredictionRequest(BaseModel):
    cyclone_id: str = Field(default="DEMO_CYCLONE")
    observations: List[TrackObservation] = Field(..., min_length=3, description="Chronological sequence (at least 3 steps)")


class ForecastHorizonPoint(BaseModel):
    horizon: str = Field(..., json_schema_extra={"example": "6h"})
    lat: float
    lon: float
    predicted_vmax_kt: float
    uncertainty_radius_km: float
    confidence_interval_pct: float
    sample_points: Optional[List[Dict[str, float]]] = None


class TrackPredictionResponse(BaseModel):
    cyclone_id: str
    current_lat: float
    current_lon: float
    current_vmax_kt: float
    forecast_horizons: List[ForecastHorizonPoint]
    model_name: str
    model_version: str
    uncertainty_method: str
    disclaimer: str = "AI-assisted decision-support research prototype. Not an official IMD warning."


# --- Risk Assessment Schemas ---
class RiskAssessmentRequest(BaseModel):
    cyclone_id: str = "DEMO_CYCLONE"
    current_vmax_kt: float = Field(..., ge=0.0, le=250.0)
    current_lat: float = Field(..., ge=-90.0, le=90.0)
    current_lon: float = Field(..., ge=-180.0, le=180.0)
    forecast_vmax_24h_kt: Optional[float] = None
    forecast_lat_24h: Optional[float] = None
    forecast_lon_24h: Optional[float] = None
    uncertainty_radius_24h_km: float = 125.0
    distance_to_coastline_km: Optional[float] = None


class RuleTraceItem(BaseModel):
    rule_id: str
    condition: str
    triggered: bool
    risk_score_impact: int
    rationale: str


class RiskAssessmentResponse(BaseModel):
    cyclone_id: str
    overall_risk_level: str = Field(..., description="Low | Moderate | High | Very High")
    total_risk_score: int = Field(..., ge=0, le=100)
    intensity_threat: str
    proximity_threat: str
    rapid_intensification_risk: bool
    rule_audit_trace: List[RuleTraceItem]
    model_version: str
    disclaimer: str = "AI-assisted decision-support research prototype. Not an official IMD warning."
