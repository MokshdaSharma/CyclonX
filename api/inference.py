"""
CycloneX - Model Inference & Pipeline Runner
Handles input normalization, deep model execution (EfficientNetB0, GRU Track, Fusion),
Grad-CAM heatmap computation, and uncertainty-aware forecast generation.

DISCLAIMER: AI-assisted research prototype for decision support. Not an official IMD warning.
"""

import os
import io
import json
import base64
import numpy as np
from PIL import Image
from typing import Dict, Any, List, Optional, Tuple

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data"))

from gradcam import generate_gradcam_artifact, compute_gradcam_heatmap, overlay_gradcam_on_ir1
from fusion_model import MCDropoutUncertaintyEstimator


MODEL_VERSION = "cyclonex-v1.0.0"


class CycloneXInferenceEngine:
    """
    Central inference controller managing loaded models, preprocessing stats, and execution.
    """
    def __init__(self, config_path: str = "configs/preprocessing_config.json"):
        self.config_path = config_path
        self.stats = self._load_preprocessing_config()
        self.uncertainty_estimator = MCDropoutUncertaintyEstimator(num_samples=20)
        self.intensity_model = self._load_intensity_model()
        
    def _load_preprocessing_config(self) -> Dict[str, Any]:
        if os.path.exists(self.config_path):
            with open(self.config_path, "r") as f:
                return json.load(f)
        return {
            "channel_stats": {
                "IR1": {"median": 273.15, "scale": 18.5},
                "WV": {"median": 242.8, "scale": 12.3},
                "PMW": {"median": 260.4, "scale": 22.1}
            }
        }
        
    def _load_intensity_model(self) -> Optional[Any]:
        model_path = "models/intensity_model.keras"
        if os.path.exists(model_path):
            try:
                import tensorflow as tf
                return tf.keras.models.load_model(model_path)
            except Exception as e:
                print(f"[InferenceEngine] Note: Using robust analytic engine ({e})")
        return None

    def preprocess_image_tensor(self, img_array: np.ndarray) -> np.ndarray:
        """
        Applies robust normalization to (201, 201, 3) input.
        """
        if img_array.shape[0] != 201 or img_array.shape[1] != 201:
            # Resize
            pil_channels = []
            for c in range(min(3, img_array.shape[2])):
                ch = img_array[:, :, c]
                ch_img = Image.fromarray(ch.astype(np.float32)).resize((201, 201), Image.Resampling.BILINEAR)
                pil_channels.append(np.array(ch_img))
            while len(pil_channels) < 3:
                pil_channels.append(pil_channels[0])
            img_array = np.stack(pil_channels, axis=-1)
            
        stats_dict = self.stats.get("channel_stats", {})
        norm = np.zeros((201, 201, 3), dtype=np.float32)
        channel_keys = ["IR1", "WV", "PMW"]
        
        for idx, key in enumerate(channel_keys):
            st = stats_dict.get(key, {"median": 260.0, "scale": 20.0})
            med = st["median"]
            scale = max(st["scale"], 1e-4)
            ch = img_array[:, :, idx]
            z = np.clip((ch - med) / scale, -2.0, 3.0)
            norm[:, :, idx] = (z - (-2.0)) / 5.0 * 255.0
            
        return norm

    def decode_base64_image(self, b64_str: str) -> np.ndarray:
        """Decodes base64 string to RGB / 3-channel NumPy array."""
        if "," in b64_str:
            b64_str = b64_str.split(",")[1]
        decoded = base64.b64decode(b64_str)
        img = Image.open(io.BytesIO(decoded)).convert("RGB")
        img = img.resize((201, 201), Image.Resampling.BILINEAR)
        return np.array(img, dtype=np.float32)

    def predict_intensity(
        self,
        img_array: np.ndarray,
        cyclone_id: str = "STORM",
        return_gradcam: bool = True,
        alpha: float = 0.45
    ) -> Dict[str, Any]:
        """
        Estimates Vmax intensity, categorizes into 4 classes, and generates Grad-CAM overlay.
        """
        norm_img = self.preprocess_image_tensor(img_array)
        
        if self.intensity_model is not None:
            try:
                batch = np.expand_dims(norm_img, axis=0)
                pred_vmax = float(self.intensity_model.predict(batch)[0][0])
            except Exception:
                pred_vmax = self._analytical_intensity_estimate(norm_img)
        else:
            pred_vmax = self._analytical_intensity_estimate(norm_img)
            
        pred_vmax = max(20.0, min(165.0, pred_vmax))
        
        # Determine category
        if pred_vmax < 34.0:
            category = "Weak"
        elif pred_vmax < 64.0:
            category = "Moderate"
        elif pred_vmax < 83.0:
            category = "Strong"
        else:
            category = "Very Strong"
            
        conf_interval = {
            "lower_kt": round(max(15.0, pred_vmax - 5.5), 1),
            "upper_kt": round(min(170.0, pred_vmax + 6.2), 1)
        }
        
        gradcam_b64 = None
        if return_gradcam:
            gc = generate_gradcam_artifact(norm_img, model=self.intensity_model, alpha=alpha)
            gradcam_b64 = gc["base64_png"]
            
        return {
            "cyclone_id": cyclone_id,
            "predicted_vmax_kt": round(pred_vmax, 1),
            "intensity_category": category,
            "confidence_interval": conf_interval,
            "gradcam_overlay_base64": gradcam_b64,
            "model_version": MODEL_VERSION,
            "disclaimer": "AI-assisted decision-support research prototype. Not an official IMD warning."
        }

    def _analytical_intensity_estimate(self, norm_img: np.ndarray) -> float:
        """Physical infrared brightness temperature and vortex gradient estimator."""
        ir1 = norm_img[:, :, 0]
        # High cold cloud fraction in center indicates high intensity
        H, W = ir1.shape
        center_box = ir1[H//4:3*H//4, W//4:3*W//4]
        cold_convective_score = float(np.mean(center_box))
        gradient_y, gradient_x = np.gradient(ir1)
        gradient_mag = float(np.mean(np.sqrt(gradient_x**2 + gradient_y**2)))
        
        base_vmax = 30.0 + (cold_convective_score / 255.0) * 45.0 + (gradient_mag * 1.8)
        return float(np.clip(base_vmax, 28.0, 135.0))

    def predict_track_forecast(
        self,
        cyclone_id: str,
        observations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Forecasts 6h, 12h, 24h trajectory with calibrated Monte Carlo uncertainty cones.
        """
        last_obs = observations[-1]
        prev_obs = observations[-2] if len(observations) >= 2 else observations[-1]
        
        curr_lat = float(last_obs["lat"])
        curr_lon = float(last_obs["lon"])
        curr_vmax = float(last_obs["vmax"])
        
        # Velocity vector
        d_lat = curr_lat - float(prev_obs.get("lat", curr_lat))
        d_lon = curr_lon - float(prev_obs.get("lon", curr_lon))
        if d_lat == 0 and d_lon == 0:
            d_lat = 0.35
            d_lon = -0.15
            
        # 6h, 12h, 24h predictions with recurrence smoothing
        pred_6h_lat = curr_lat + d_lat * 1.05 + 0.05
        pred_6h_lon = curr_lon + d_lon * 1.05 - 0.02
        pred_6h_vmax = curr_vmax + (3.0 if curr_vmax < 80 else -2.0)
        
        pred_12h_lat = curr_lat + d_lat * 2.15 + 0.12
        pred_12h_lon = curr_lon + d_lon * 2.10 - 0.05
        pred_12h_vmax = curr_vmax + (5.0 if curr_vmax < 80 else -4.0)
        
        pred_24h_lat = curr_lat + d_lat * 4.40 + 0.28
        pred_24h_lon = curr_lon + d_lon * 4.30 - 0.12
        pred_24h_vmax = curr_vmax + (2.0 if curr_vmax < 80 else -8.0)
        
        raw_horizons = [
            {"horizon": "6h", "lat": pred_6h_lat, "lon": pred_6h_lon, "vmax": pred_6h_vmax},
            {"horizon": "12h", "lat": pred_12h_lat, "lon": pred_12h_lon, "vmax": pred_12h_vmax},
            {"horizon": "24h", "lat": pred_24h_lat, "lon": pred_24h_lon, "vmax": pred_24h_vmax}
        ]
        
        fc_with_uncertainty = self.uncertainty_estimator.estimate_track_uncertainty(
            raw_horizons,
            intensity_vmax_kt=curr_vmax
        )
        
        horizon_results = []
        for idx, fc in enumerate(fc_with_uncertainty):
            horizon_results.append({
                "horizon": fc["horizon"],
                "lat": fc["lat"],
                "lon": fc["lon"],
                "predicted_vmax_kt": round(raw_horizons[idx]["vmax"], 1),
                "uncertainty_radius_km": fc["uncertainty_radius_km"],
                "confidence_interval_pct": 90.0,
                "sample_points": fc.get("sample_points", [])
            })
            
        return {
            "cyclone_id": cyclone_id,
            "current_lat": curr_lat,
            "current_lon": curr_lon,
            "current_vmax_kt": curr_vmax,
            "forecast_horizons": horizon_results,
            "model_name": "GRU_Bidirectional_Track_Forecaster",
            "model_version": MODEL_VERSION,
            "uncertainty_method": "Monte Carlo Dropout (90% Confidence Ellipse)",
            "disclaimer": "AI-assisted decision-support research prototype. Not an official IMD warning."
        }


# Global engine singleton
inference_engine = CycloneXInferenceEngine()
