"""
CycloneX - FastAPI Endpoint Integration Tests
Validates /, /health, /predict, /predict-track, /predict-risk, response schemas, and error statuses.
"""

import os
import sys
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "api"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data"))

from app import app


client = TestClient(app)


def test_system_info():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "CycloneX" in data["system_name"]
    assert "model_version" in data
    assert "disclaimer" in data


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "model_version" in data
    assert "disclaimer" in data


def test_demo_cyclones_endpoint():
    response = client.get("/demo-cyclones")
    assert response.status_code == 200
    data = response.json()
    assert "FANI_2019" in data["cyclones"]
    assert "AMPHAN_2020" in data["cyclones"]


def test_predict_intensity_default_payload():
    response = client.post("/predict", json={"cyclone_id": "TEST_FANI", "return_gradcam": True})
    assert response.status_code == 200
    data = response.json()
    assert data["cyclone_id"] == "TEST_FANI"
    assert "predicted_vmax_kt" in data
    assert data["intensity_category"] in ["Weak", "Moderate", "Strong", "Very Strong"]
    assert "model_version" in data
    assert "disclaimer" in data
    assert data["gradcam_overlay_base64"] is not None
    assert data["gradcam_overlay_base64"].startswith("data:image/png;base64,")


def test_predict_track_forecast():
    payload = {
        "cyclone_id": "TEST_STORM",
        "observations": [
            {"timestamp": "2023-05-10 00:00 UTC", "lat": 10.0, "lon": 85.0, "vmax": 40.0, "pres": 998.0},
            {"timestamp": "2023-05-10 06:00 UTC", "lat": 10.5, "lon": 84.8, "vmax": 50.0, "pres": 992.0},
            {"timestamp": "2023-05-10 12:00 UTC", "lat": 11.2, "lon": 84.5, "vmax": 65.0, "pres": 985.0}
        ]
    }
    response = client.post("/predict-track", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert len(data["forecast_horizons"]) == 3
    horizons = [h["horizon"] for h in data["forecast_horizons"]]
    assert horizons == ["6h", "12h", "24h"]
    assert data["forecast_horizons"][2]["uncertainty_radius_km"] > data["forecast_horizons"][0]["uncertainty_radius_km"]


def test_predict_track_insufficient_history_returns_422():
    payload = {
        "cyclone_id": "SHORT_STORM",
        "observations": [
            {"lat": 10.0, "lon": 85.0, "vmax": 40.0}
        ]
    }
    response = client.post("/predict-track", json=payload)
    assert response.status_code == 422


def test_predict_risk_rule_evaluation():
    payload = {
        "cyclone_id": "TEST_SEVERE_STORM",
        "current_vmax_kt": 95.0,
        "current_lat": 19.5,
        "current_lon": 86.0,
        "forecast_vmax_24h_kt": 120.0,
        "distance_to_coastline_km": 50.0
    }
    response = client.post("/predict-risk", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["overall_risk_level"] in ["High", "Very High"]
    assert data["total_risk_score"] >= 70
    assert data["rapid_intensification_risk"] is True
    assert len(data["rule_audit_trace"]) >= 3
    assert "disclaimer" in data
