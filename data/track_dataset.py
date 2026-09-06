"""
CycloneX - Best-Track Sequence Dataset & Geospatial Helpers
Ingests IMD/RSMC or JTWC best-track records (lat, lon, max wind, pressure, timestamp),
computes motion dynamics (deltas, speed, heading),
and constructs sliding-window sequences for LSTM/GRU temporal track & intensity forecasting.

DISCLAIMER: AI-assisted research prototype for decision support. Not an official IMD warning.
"""

import math
import os
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Any, Optional


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Computes the great-circle distance between two points on Earth in kilometers
    using the Haversine formula.
    """
    R = 6371.0 # Earth radius in kilometers
    
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    
    a = (math.sin(delta_phi / 2.0) ** 2 +
         math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2)
    
    # Avoid numerical precision errors resulting in a > 1.0
    a = min(1.0, max(0.0, a))
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    
    return R * c


def compute_track_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds motion features to chronological cyclone records:
    - d_lat (degrees / 6h)
    - d_lon (degrees / 6h)
    - speed_kmh (km / h)
    - heading_deg (0 - 360 azimuth)
    - d_vmax (vmax rate of change, kt / 6h)
    - d_pres (pressure rate of change, hPa / 6h)
    """
    df = df.sort_values(by="timestamp").copy()
    
    # Differences
    df["d_lat"] = df["lat"].diff().fillna(0.0)
    df["d_lon"] = df["lon"].diff().fillna(0.0)
    df["d_vmax"] = df["vmax"].diff().fillna(0.0)
    df["d_pres"] = df["pres"].diff().fillna(0.0)
    
    # Distance between consecutive 6h points
    distances_km = [0.0]
    headings = [0.0]
    
    lat_vals = df["lat"].values
    lon_vals = df["lon"].values
    
    for i in range(1, len(lat_vals)):
        d_km = haversine_distance(lat_vals[i-1], lon_vals[i-1], lat_vals[i], lon_vals[i])
        distances_km.append(d_km)
        
        # Calculate bearing / heading
        d_lon_rad = math.radians(lon_vals[i] - lon_vals[i-1])
        lat1_rad = math.radians(lat_vals[i-1])
        lat2_rad = math.radians(lat_vals[i])
        
        y = math.sin(d_lon_rad) * math.cos(lat2_rad)
        x = (math.cos(lat1_rad) * math.sin(lat2_rad) -
             math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(d_lon_rad))
        bearing = (math.degrees(math.atan2(y, x)) + 360) % 360
        headings.append(bearing)
        
    df["dist_step_km"] = distances_km
    # Assuming 6-hour standard observation synoptic intervals
    df["speed_kmh"] = df["dist_step_km"] / 6.0
    df["heading_deg"] = headings
    
    return df


def build_sliding_windows(
    cyclone_tracks: pd.DataFrame,
    window_size: int = 8,
    forecast_steps: List[int] = [1, 2, 4], # [6h, 12h, 24h] assuming 6h step
    feature_cols: Optional[List[str]] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Converts multi-cyclone track history into fixed-length sliding sequences:
    Inputs X: shape (N_samples, window_size, len(feature_cols))
    Targets Y_track: future coordinates [lat_6h, lon_6h, lat_12h, lon_12h, lat_24h, lon_24h]
    Targets Y_intensity: future vmax/pres [vmax_6h, pres_6h, vmax_12h, pres_12h, vmax_24h, pres_24h]
    cyclone_ids: list of storm IDs per sequence (for leak-free splitting)
    """
    if feature_cols is None:
        feature_cols = ["lat", "lon", "vmax", "pres", "d_lat", "d_lon", "speed_kmh", "heading_deg"]
        
    max_forecast_step = max(forecast_steps)
    
    x_list = []
    y_track_list = []
    y_intensity_list = []
    c_ids_list = []
    
    for c_id, group in cyclone_tracks.groupby("cyclone_id"):
        df_storm = compute_track_features(group)
        if len(df_storm) < window_size + max_forecast_step:
            continue
            
        records = df_storm[feature_cols].values
        lat_arr = df_storm["lat"].values
        lon_arr = df_storm["lon"].values
        vmax_arr = df_storm["vmax"].values
        pres_arr = df_storm["pres"].values
        
        for t in range(len(df_storm) - window_size - max_forecast_step + 1):
            x_seq = records[t:t + window_size]
            
            # Extract target coordinates for +6h (t+1), +12h (t+2), +24h (t+4)
            y_track = []
            y_intensity = []
            for step in forecast_steps:
                target_idx = t + window_size - 1 + step
                y_track.extend([lat_arr[target_idx], lon_arr[target_idx]])
                y_intensity.extend([vmax_arr[target_idx], pres_arr[target_idx]])
                
            x_list.append(x_seq)
            y_track_list.append(y_track)
            y_intensity_list.append(y_intensity)
            c_ids_list.append(c_id)
            
    if not x_list:
        return np.empty((0, window_size, len(feature_cols))), np.empty((0, len(forecast_steps)*2)), np.empty((0, len(forecast_steps)*2)), []
        
    return (
        np.array(x_list, dtype=np.float32),
        np.array(y_track_list, dtype=np.float32),
        np.array(y_intensity_list, dtype=np.float32),
        c_ids_list
    )


def generate_sample_best_tracks(output_csv: str = "data/sample_best_tracks.csv", num_storms: int = 15) -> pd.DataFrame:
    """
    Generates realistic historical best-track sequences for Indian Ocean / Bay of Bengal & Arabian Sea cyclones.
    """
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    storms_info = [
        ("FANI_2019", 9.5, 88.5, 30.0, 1002.0, 28, 0.45, -0.15),
        ("AMPHAN_2020", 10.8, 86.5, 35.0, 998.0, 26, 0.50, 0.05),
        ("BIPARJOY_2023", 12.0, 66.0, 35.0, 1000.0, 32, 0.35, 0.10),
        ("TAUKTAE_2021", 10.5, 73.0, 35.0, 1000.0, 24, 0.48, -0.05),
        ("HUDHUD_2014", 12.5, 92.5, 35.0, 1000.0, 22, 0.38, -0.45),
        ("VARDAH_2016", 11.2, 91.0, 30.0, 1004.0, 20, 0.20, -0.55),
        ("TITLI_2018", 14.0, 87.0, 35.0, 1000.0, 18, 0.40, -0.15),
        ("YAAS_2021", 16.0, 89.5, 35.0, 998.0, 20, 0.42, -0.22),
        ("GULAB_2021", 17.5, 90.0, 30.0, 1004.0, 16, 0.15, -0.60),
        ("MOCHA_2023", 11.0, 88.0, 35.0, 1000.0, 26, 0.52, 0.25),
        ("NIVAR_2020", 10.0, 83.5, 30.0, 1004.0, 18, 0.25, -0.30),
        ("BUREVI_2020", 8.0, 84.0, 30.0, 1004.0, 18, 0.15, -0.40),
        ("NISARGA_2020", 14.0, 71.5, 30.0, 1002.0, 18, 0.45, 0.15),
        ("VAYU_2019", 14.5, 70.5, 35.0, 1000.0, 24, 0.35, -0.08),
        ("BULBUL_2019", 13.0, 89.0, 35.0, 1000.0, 22, 0.45, 0.05)
    ]
    
    records = []
    for s_idx in range(min(num_storms, len(storms_info))):
        s_name, start_lat, start_lon, start_vmax, start_pres, steps, lat_step, lon_step = storms_info[s_idx]
        
        curr_lat = start_lat
        curr_lon = start_lon
        curr_vmax = start_vmax
        curr_pres = start_pres
        
        peak_step = steps // 2
        for t in range(steps):
            # Lifecycle curve
            if t <= peak_step:
                intensity_delta = np.random.uniform(4.0, 9.0)
            else:
                intensity_delta = -np.random.uniform(4.0, 8.0)
                
            curr_vmax = float(np.clip(curr_vmax + intensity_delta, 25.0, 145.0))
            curr_pres = float(np.clip(1012.0 - (curr_vmax * 0.72) + np.random.normal(0, 1.5), 900.0, 1014.0))
            curr_lat = float(curr_lat + lat_step + np.random.normal(0, 0.04))
            curr_lon = float(curr_lon + lon_step + np.random.normal(0, 0.04))
            
            records.append({
                "cyclone_id": s_name,
                "timestamp": f"202305{10 + (t // 4):02d}_{(t % 4)*6:02d}00",
                "lat": round(curr_lat, 2),
                "lon": round(curr_lon, 2),
                "vmax": round(curr_vmax, 1),
                "pres": round(curr_pres, 1)
            })
            
    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False)
    print(f"Saved {len(df)} best-track records across {len(storms_info)} cyclones to {output_csv}")
    return df


if __name__ == "__main__":
    df = generate_sample_best_tracks()
    X, Y_tr, Y_int, c_ids = build_sliding_windows(df)
    print(f"Generated sequences: X={X.shape}, Y_track={Y_tr.shape}, Y_intensity={Y_int.shape}")
