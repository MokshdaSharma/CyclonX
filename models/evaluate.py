"""
CycloneX - Comprehensive Evaluation Metrics Suite
Computes regression (MAE, RMSE, R2, Bias), classification (Accuracy, Precision, Recall, F1, Confusion Matrix),
and geospatial (Haversine Track Distance in km) metrics.

DISCLAIMER: AI-assisted research prototype for decision support. Not an official IMD warning.
"""

import json
import math
import numpy as np
from typing import Dict, Any, List, Tuple
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, f1_score, precision_score, recall_score, confusion_matrix


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculates great-circle distance between two coordinates in km."""
    R = 6371.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    
    a = math.sin(dp / 2.0)**2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2.0)**2
    a = min(1.0, max(0.0, a))
    return R * 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Computes MAE, RMSE, R2 score, and Mean Bias Error."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    bias = float(np.mean(y_pred - y_true))
    
    return {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "r2": round(r2, 4),
        "mean_bias": round(bias, 4)
    }


def compute_classification_metrics(
    y_true_vmax: np.ndarray,
    y_pred_vmax: np.ndarray,
    categories: List[str] = ["Weak", "Moderate", "Strong", "Very Strong"]
) -> Dict[str, Any]:
    """
    Categorizes Vmax into 4 classes and computes classification performance.
    """
    def categorize(v):
        if v < 34.0:
            return 0 # Weak
        elif v < 64.0:
            return 1 # Moderate
        elif v < 83.0:
            return 2 # Strong
        else:
            return 3 # Very Strong
            
    c_true = np.array([categorize(v) for v in y_true_vmax])
    c_pred = np.array([categorize(v) for v in y_pred_vmax])
    
    acc = float(np.mean(c_true == c_pred))
    prec_macro = float(precision_score(c_true, c_pred, average="macro", zero_division=0))
    rec_macro = float(recall_score(c_true, c_pred, average="macro", zero_division=0))
    f1_macro = float(f1_score(c_true, c_pred, average="macro", zero_division=0))
    f1_weighted = float(f1_score(c_true, c_pred, average="weighted", zero_division=0))
    
    cm = confusion_matrix(c_true, c_pred, labels=[0, 1, 2, 3]).tolist()
    
    # Per-class breakdown
    per_class = {}
    for idx, name in enumerate(categories):
        mask = (c_true == idx)
        if np.sum(mask) > 0:
            sub_true = y_true_vmax[mask]
            sub_pred = y_pred_vmax[mask]
            per_class[name] = {
                "count": int(np.sum(mask)),
                "mae": round(float(mean_absolute_error(sub_true, sub_pred)), 2),
                "rmse": round(float(np.sqrt(mean_squared_error(sub_true, sub_pred))), 2)
            }
        else:
            per_class[name] = {"count": 0, "mae": 0.0, "rmse": 0.0}
            
    return {
        "accuracy": round(acc, 4),
        "precision_macro": round(prec_macro, 4),
        "recall_macro": round(rec_macro, 4),
        "f1_macro": round(f1_macro, 4),
        "f1_weighted": round(f1_weighted, 4),
        "confusion_matrix": cm,
        "per_class_breakdown": per_class
    }


def compute_track_haversine_metrics(
    y_true_coords: np.ndarray,
    y_pred_coords: np.ndarray,
    horizons: List[str] = ["6h", "12h", "24h"]
) -> Dict[str, Dict[str, float]]:
    """
    Computes Haversine distance error (in km) per forecast horizon.
    Coordinates shape: (N, len(horizons)*2) -> [lat1, lon1, lat2, lon2, lat3, lon3]
    """
    results = {}
    n_horizons = len(horizons)
    
    for h_idx, h_name in enumerate(horizons):
        lat_idx = h_idx * 2
        lon_idx = h_idx * 2 + 1
        
        errors = []
        for i in range(len(y_true_coords)):
            t_lat, t_lon = y_true_coords[i, lat_idx], y_true_coords[i, lon_idx]
            p_lat, p_lon = y_pred_coords[i, lat_idx], y_pred_coords[i, lon_idx]
            dist_km = haversine_km(t_lat, t_lon, p_lat, p_lon)
            errors.append(dist_km)
            
        errors = np.array(errors)
        results[h_name] = {
            "mean_haversine_km": round(float(np.mean(errors)), 2),
            "median_haversine_km": round(float(np.median(errors)), 2),
            "p90_haversine_km": round(float(np.percentile(errors, 90)), 2),
            "min_km": round(float(np.min(errors)), 2),
            "max_km": round(float(np.max(errors)), 2)
        }
        
    return results


def evaluate_intensity_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: str = "models/model_performance.json"
) -> Dict[str, Any]:
    """Generates and saves complete intensity model performance report."""
    reg_metrics = compute_regression_metrics(y_true, y_pred)
    cls_metrics = compute_classification_metrics(y_true, y_pred)
    
    report = {
        "model_name": "EfficientNetB0_Intensity_Regressor",
        "task": "Tropical Cyclone Vmax (Knots) Estimation",
        "sample_count": len(y_true),
        "regression": reg_metrics,
        "classification": cls_metrics,
        "disclaimer": "AI-assisted research prototype. Not an official IMD warning."
    }
    
    with open(save_path, "w") as f:
        json.dump(report, f, indent=2)
        
    print(f"Model performance report successfully saved to: {save_path}")
    return report
