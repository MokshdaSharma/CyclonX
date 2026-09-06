"""
CycloneX - Model & Explainability Unit Tests
Validates Haversine calculation, Grad-CAM output dimensions, and risk score boundaries.
"""

import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "api"))

from evaluate import haversine_km, compute_regression_metrics, compute_classification_metrics
from gradcam import generate_gradcam_artifact, compute_gradcam_heatmap
from track_dataset import haversine_distance
from train_track_model import persistence_baseline_track_forecast
from risk import evaluate_cyclone_risk


def test_haversine_distance_accuracy():
    # Distance between Chennai (13.0827, 80.2707) and Kolkata (22.5726, 88.3639) is ~1366 km
    dist = haversine_distance(13.0827, 80.2707, 22.5726, 88.3639)
    assert 1350.0 <= dist <= 1380.0


def test_gradcam_heatmap_dimensions():
    sample_img = np.random.uniform(200, 300, (201, 201, 3)).astype(np.float32)
    artifact = generate_gradcam_artifact(sample_img, model=None, alpha=0.5)
    
    assert artifact["heatmap"].shape == (201, 201)
    assert artifact["blended_rgb"].shape == (201, 201, 3)
    assert artifact["base64_png"].startswith("data:image/png;base64,")


def test_persistence_baseline_forecast():
    # 1 sample, 8 timesteps, 8 features
    X_test = np.zeros((1, 8, 8), dtype=np.float32)
    X_test[0, -1, 0] = 12.0  # lat
    X_test[0, -1, 1] = 85.0  # lon
    X_test[0, -1, 4] = 0.5   # d_lat
    X_test[0, -1, 5] = -0.2  # d_lon
    
    preds = persistence_baseline_track_forecast(X_test, horizons_steps=[1, 2, 4])
    # 6h: 12 + 0.5 = 12.5, 85 - 0.2 = 84.8
    assert np.isclose(preds[0, 0], 12.5)
    assert np.isclose(preds[0, 1], 84.8)
    # 24h (step 4): 12 + 2.0 = 14.0, 85 - 0.8 = 84.2
    assert np.isclose(preds[0, 4], 14.0)
    assert np.isclose(preds[0, 5], 84.2)


def test_risk_evaluation_score_bounds():
    risk_low = evaluate_cyclone_risk(
        cyclone_id="WEAK_STORM",
        current_vmax_kt=25.0,
        current_lat=10.0,
        current_lon=90.0,
        distance_to_coastline_km=600.0
    )
    assert risk_low.overall_risk_level == "Low"
    assert 0 <= risk_low.total_risk_score <= 30
    
    risk_high = evaluate_cyclone_risk(
        cyclone_id="SUPER_STORM",
        current_vmax_kt=120.0,
        current_lat=19.0,
        current_lon=86.0,
        forecast_vmax_24h_kt=145.0,
        distance_to_coastline_km=40.0
    )
    assert risk_high.overall_risk_level == "Very High"
    assert risk_high.total_risk_score >= 80
    assert risk_high.rapid_intensification_risk is True
