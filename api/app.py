"""
CycloneX - FastAPI RESTful Backend
Endpoints:
- GET /               : System info & metadata
- GET /health         : Health status and model version
- GET /demo-cyclones  : Preloaded Indian Ocean cyclone profiles (Fani, Amphan, Biparjoy, Mocha)
- POST /predict       : Intensity estimation & Grad-CAM visual explainability
- POST /predict-track : 6h / 12h / 24h trajectory forecast with uncertainty cone radii
- POST /predict-risk  : Transparent rule-based risk evaluation matrix

DISCLAIMER: AI-assisted research prototype for decision support. Not an official IMD warning.
"""

import os
import json
import numpy as np
from fastapi import FastAPI, HTTPException, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Dict, Any, List

import os
import sys

# Ensure local api folder is at the top of sys.path
api_dir = os.path.dirname(os.path.abspath(__file__))
if api_dir not in sys.path:
    sys.path.insert(0, api_dir)

try:
    from api.schemas import (
        HealthResponse,
        SystemInfoResponse,
        SatelliteFrameRequest,
        IntensityPredictionResponse,
        TrackPredictionRequest,
        TrackPredictionResponse,
        RiskAssessmentRequest,
        RiskAssessmentResponse
    )
    from api.inference import inference_engine, MODEL_VERSION
    from api.risk import evaluate_cyclone_risk
except ImportError:
    from schemas import (
        HealthResponse,
        SystemInfoResponse,
        SatelliteFrameRequest,
        IntensityPredictionResponse,
        TrackPredictionRequest,
        TrackPredictionResponse,
        RiskAssessmentRequest,
        RiskAssessmentResponse
    )
    from inference import inference_engine, MODEL_VERSION
    from risk import evaluate_cyclone_risk


app = FastAPI(
    title="CycloneX Decision-Support API",
    description="AI/ML System for Tropical Cyclone Identification, Classification, Track Forecasting & Grad-CAM Explainability (SIH PS 26070 - MoES/IMD).",
    version=MODEL_VERSION,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware for local frontend development and production hosting (Vercel/Spaces)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_model=SystemInfoResponse)
def get_system_info():
    """Returns general metadata, active models, supported channels, and disclaimer."""
    return SystemInfoResponse(
        system_name="CycloneX Decision-Support AI System",
        problem_statement="SIH 26070 (Ministry of Earth Sciences / IMD)",
        model_version=MODEL_VERSION,
        endpoints=["/health", "/predict", "/predict-track", "/predict-risk", "/demo-cyclones"],
        intensity_classes=["Weak (<34 kt)", "Moderate (34-63 kt)", "Strong (64-82 kt)", "Very Strong (>=83 kt)"],
        channels_supported=["IR1 (Infrared)", "WV (Water Vapor)", "PMW (Passive Microwave)"],
        disclaimer="AI-assisted research prototype. Not an official IMD warning."
    )


@app.get("/health", response_model=HealthResponse)
def health_check():
    """Liveness and readiness health probe."""
    return HealthResponse(
        status="healthy",
        model_version=MODEL_VERSION,
        active_models=["EfficientNetB0_Intensity", "GRU_Track_Forecaster", "Multimodal_Fusion_MCDropout"],
        disclaimer="AI-assisted research prototype. Not an official IMD warning."
    )


@app.get("/demo-cyclones")
def get_demo_cyclones() -> Dict[str, Any]:
    """
    Returns curated historical cyclones with satellite patterns and best-track observations
    for instant frontend demonstration and benchmark testing.
    """
    demo_data = {
        "FANI_2019": {
            "name": "Extremely Severe Cyclonic Storm Fani (2019)",
            "basin": "Bay of Bengal",
            "current_vmax_kt": 115.0,
            "category": "Very Strong",
            "history": [
                {"timestamp": "2019-04-29 00:00 UTC", "lat": 8.5, "lon": 87.0, "vmax": 45.0, "pres": 996.0},
                {"timestamp": "2019-04-29 12:00 UTC", "lat": 9.2, "lon": 86.8, "vmax": 55.0, "pres": 990.0},
                {"timestamp": "2019-04-30 00:00 UTC", "lat": 10.3, "lon": 86.2, "vmax": 75.0, "pres": 978.0},
                {"timestamp": "2019-04-30 12:00 UTC", "lat": 11.6, "lon": 85.5, "vmax": 95.0, "pres": 962.0},
                {"timestamp": "2019-05-01 00:00 UTC", "lat": 13.1, "lon": 84.8, "vmax": 105.0, "pres": 950.0},
                {"timestamp": "2019-05-01 12:00 UTC", "lat": 14.8, "lon": 84.2, "vmax": 115.0, "pres": 938.0}
            ]
        },
        "AMPHAN_2020": {
            "name": "Super Cyclonic Storm Amphan (2020)",
            "basin": "Bay of Bengal",
            "current_vmax_kt": 130.0,
            "category": "Very Strong",
            "history": [
                {"timestamp": "2020-05-16 12:00 UTC", "lat": 10.8, "lon": 86.5, "vmax": 40.0, "pres": 998.0},
                {"timestamp": "2020-05-17 00:00 UTC", "lat": 11.5, "lon": 86.2, "vmax": 55.0, "pres": 988.0},
                {"timestamp": "2020-05-17 12:00 UTC", "lat": 12.5, "lon": 86.4, "vmax": 75.0, "pres": 972.0},
                {"timestamp": "2020-05-18 00:00 UTC", "lat": 13.4, "lon": 86.3, "vmax": 115.0, "pres": 935.0},
                {"timestamp": "2020-05-18 12:00 UTC", "lat": 14.2, "lon": 86.3, "vmax": 130.0, "pres": 920.0}
            ]
        },
        "BIPARJOY_2023": {
            "name": "Very Severe Cyclonic Storm Biparjoy (2023)",
            "basin": "Arabian Sea",
            "current_vmax_kt": 75.0,
            "category": "Strong",
            "history": [
                {"timestamp": "2023-06-08 00:00 UTC", "lat": 13.5, "lon": 66.2, "vmax": 50.0, "pres": 990.0},
                {"timestamp": "2023-06-08 12:00 UTC", "lat": 14.2, "lon": 66.0, "vmax": 65.0, "pres": 980.0},
                {"timestamp": "2023-06-09 00:00 UTC", "lat": 15.0, "lon": 66.1, "vmax": 75.0, "pres": 974.0},
                {"timestamp": "2023-06-09 12:00 UTC", "lat": 16.0, "lon": 66.3, "vmax": 80.0, "pres": 970.0},
                {"timestamp": "2023-06-10 00:00 UTC", "lat": 17.2, "lon": 67.2, "vmax": 75.0, "pres": 975.0}
            ]
        },
        "MOCHA_2023": {
            "name": "Extremely Severe Cyclonic Storm Mocha (2023)",
            "basin": "Bay of Bengal",
            "current_vmax_kt": 120.0,
            "category": "Very Strong",
            "history": [
                {"timestamp": "2023-05-11 06:00 UTC", "lat": 11.2, "lon": 88.2, "vmax": 45.0, "pres": 994.0},
                {"timestamp": "2023-05-11 18:00 UTC", "lat": 12.1, "lon": 87.8, "vmax": 60.0, "pres": 984.0},
                {"timestamp": "2023-05-12 06:00 UTC", "lat": 13.2, "lon": 88.0, "vmax": 80.0, "pres": 968.0},
                {"timestamp": "2023-05-12 18:00 UTC", "lat": 14.6, "lon": 88.7, "vmax": 105.0, "pres": 948.0},
                {"timestamp": "2023-05-13 06:00 UTC", "lat": 16.2, "lon": 90.0, "vmax": 120.0, "pres": 932.0}
            ]
        }
    }
    return {
        "count": len(demo_data),
        "cyclones": demo_data,
        "disclaimer": "AI-assisted decision-support research prototype. Not an official IMD warning."
    }


@app.post("/predict", response_model=IntensityPredictionResponse)
def predict_satellite_intensity(req: SatelliteFrameRequest):
    """
    Processes a multi-channel satellite frame (IR1, WV, PMW), outputs estimated Vmax (knots),
    intensity classification, and generates Grad-CAM explainability heatmap overlay.
    """
    try:
        if req.image_base64 is not None:
            img_array = inference_engine.decode_base64_image(req.image_base64)
        elif req.raw_array is not None:
            img_array = np.array(req.raw_array, dtype=np.float32)
            if img_array.ndim != 3 or img_array.shape[2] < 3:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="raw_array must be a 3D matrix with at least 3 channels (H, W, C >= 3)"
                )
        else:
            # Generate representative synthetic TCIR storm frame if no payload passed
            x = np.linspace(-3, 3, 201)
            y = np.linspace(-3, 3, 201)
            xx, yy = np.meshgrid(x, y)
            r = np.sqrt(xx**2 + yy**2)
            ir1 = 280.0 - 70.0 * np.exp(-0.5 * (r - 0.3)**2)
            wv = 245.0 - 35.0 * np.exp(-0.3 * (r - 0.5)**2)
            pmw = 265.0 - 50.0 * np.exp(-0.8 * (r - 0.2)**2)
            img_array = np.stack([ir1, wv, pmw], axis=-1)
            
        result = inference_engine.predict_intensity(
            img_array=img_array,
            cyclone_id=req.cyclone_id,
            return_gradcam=req.return_gradcam,
            alpha=req.gradcam_alpha
        )
        return IntensityPredictionResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to process satellite image: {str(e)}"
        )


@app.post("/predict-track", response_model=TrackPredictionResponse)
def predict_cyclone_track(req: TrackPredictionRequest):
    """
    Accepts sequence of historical observations and forecasts trajectory coordinates
    at 6h, 12h, and 24h horizons with calibrated 90% confidence uncertainty radii.
    """
    if len(req.observations) < 2:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="At least 2 chronological track observations are required to determine storm kinematics."
        )
        
    obs_dicts = [obs.model_dump() for obs in req.observations]
    res = inference_engine.predict_track_forecast(req.cyclone_id, obs_dicts)
    return TrackPredictionResponse(**res)


@app.post("/predict-risk", response_model=RiskAssessmentResponse)
def predict_risk_level(req: RiskAssessmentRequest):
    """
    Evaluates multi-factor decision-support risk score (0-100) and transparent rule audit trail.
    """
    assessment = evaluate_cyclone_risk(
        cyclone_id=req.cyclone_id,
        current_vmax_kt=req.current_vmax_kt,
        current_lat=req.current_lat,
        current_lon=req.current_lon,
        forecast_vmax_24h_kt=req.forecast_vmax_24h_kt,
        forecast_lat_24h=req.forecast_lat_24h,
        forecast_lon_24h=req.forecast_lon_24h,
        uncertainty_radius_24h_km=req.uncertainty_radius_24h_km,
        distance_to_coastline_km=req.distance_to_coastline_km,
        model_version=MODEL_VERSION
    )
    return assessment
