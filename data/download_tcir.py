"""
CycloneX - TCIR-2017 Dataset Ingestion and Preparation
Downloads or generates synthetic/sample subsets of the Tropical Cyclone Image Dataset (TCIR-2017).
Extracts channels IR1, WV, PMW at 201x201 resolution, dropping VIS due to high missing data rates.

DISCLAIMER: AI-assisted research prototype for decision support. Not an official IMD warning.
"""

import os
import argparse
import numpy as np
import h5py
import pandas as pd
from typing import Tuple, List, Dict, Any

# Primary remote sources for TCIR-2017 subset
DEFAULT_TCIR_URL = "http://tcir.csie.ntu.edu.tw/tcir/TCIR-ALL.h5"


def create_sample_tcir_dataset(
    output_path: str,
    num_cyclones: int = 10,
    frames_per_cyclone: int = 8,
    img_size: int = 201
) -> str:
    """
    Generates a realistic synthetic TCIR-2017 HDF5 dataset for testing, CI/CD, and local offline dev.
    Includes IR1 (Infrared), WV (Water Vapor), PMW (Passive Microwave), and VIS (Visible) channels.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    cyclone_names = [
        "FANI_2019", "AMPHAN_2020", "BIPARJOY_2023", "TAUKTAE_2021", "HUDHUD_2014",
        "VARDAH_2016", "TITLI_2018", "YAAS_2021", "GULAB_2021", "MOCHA_2023"
    ]
    
    total_frames = num_cyclones * frames_per_cyclone
    print(f"Generating synthetic TCIR-2017 dataset at: {output_path} with {total_frames} frames ({num_cyclones} cyclones)...")
    
    with h5py.File(output_path, "w") as f:
        # Create groups or top-level datasets
        # TCIR structure typically contains matrix data: shape (N, 201, 201, 4) -> [IR1, WV, PMW, VIS]
        matrix_ds = f.create_dataset(
            "matrix",
            shape=(total_frames, img_size, img_size, 4),
            dtype="float32",
            chunks=True
        )
        
        # Meta info
        cyclone_ids = []
        timestamps = []
        vmax_list = []
        pm_list = []
        lat_list = []
        lon_list = []
        
        frame_idx = 0
        for c_idx in range(num_cyclones):
            c_name = cyclone_names[c_idx % len(cyclone_names)]
            base_vmax = np.random.uniform(35, 110)
            base_lat = np.random.uniform(10.0, 18.0)
            base_lon = np.random.uniform(80.0, 92.0)
            base_pres = 1010.0 - (base_vmax * 0.7)
            
            for t in range(frames_per_cyclone):
                # Simulate eye/vortex pattern in IR1, WV, PMW
                x = np.linspace(-3, 3, img_size)
                y = np.linspace(-3, 3, img_size)
                xx, yy = np.meshgrid(x, y)
                r = np.sqrt(xx**2 + yy**2)
                
                # IR1: Cloud brightness temp in Kelvin (200-300K)
                ir1 = 280.0 - 65.0 * np.exp(-0.5 * (r - 0.4)**2) + np.random.normal(0, 3, (img_size, img_size))
                # WV: Upper tropospheric moisture (220-270K)
                wv = 250.0 - 30.0 * np.exp(-0.3 * (r - 0.6)**2) + np.random.normal(0, 2, (img_size, img_size))
                # PMW: Deep convective rainbands (240-290K)
                pmw = 270.0 - 45.0 * np.exp(-0.8 * (r - 0.3)**2) + np.random.normal(0, 4, (img_size, img_size))
                # VIS: Daytime only / missing at night (simulated missingness with NaNs/zeros)
                vis = np.clip(np.random.uniform(0, 1, (img_size, img_size)), 0, 1)
                if t % 2 == 1:
                    vis[:] = 0.0 # Simulate nighttime drop
                
                frame_data = np.stack([ir1, wv, pmw, vis], axis=-1).astype(np.float32)
                matrix_ds[frame_idx] = frame_data
                
                cyclone_ids.append(c_name)
                timestamps.append(f"202305{10+t:02d}_{t*6:02d}00")
                vmax_list.append(float(np.clip(base_vmax + np.sin(t / 2.0) * 10.0 + np.random.normal(0, 2), 25, 140)))
                pm_list.append(float(base_pres - np.sin(t / 2.0) * 8.0 + np.random.normal(0, 1)))
                lat_list.append(float(base_lat + t * 0.4 + np.random.normal(0, 0.05)))
                lon_list.append(float(base_lon - t * 0.2 + np.random.normal(0, 0.05)))
                
                frame_idx += 1
                
        # Store metadata attributes / datasets
        f.create_dataset("info/cyclone_id", data=np.array(cyclone_ids, dtype=h5py.string_dtype()))
        f.create_dataset("info/time", data=np.array(timestamps, dtype=h5py.string_dtype()))
        f.create_dataset("info/vmax", data=np.array(vmax_list, dtype="float32"))
        f.create_dataset("info/pres", data=np.array(pm_list, dtype="float32"))
        f.create_dataset("info/lat", data=np.array(lat_list, dtype="float32"))
        f.create_dataset("info/lon", data=np.array(lon_list, dtype="float32"))
        
    print(f"Dataset successfully created with {total_frames} records at {output_path}")
    return output_path


def load_raw_tcir(h5_path: str) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Loads raw TCIR HDF5 dataset, dropping VIS channel (channel index 3),
    retaining [IR1, WV, PMW] at indices [0, 1, 2].
    """
    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"TCIR dataset file not found at: {h5_path}")
        
    with h5py.File(h5_path, "r") as f:
        # Load all matrix frames: shape (N, 201, 201, 4)
        raw_matrix = f["matrix"][:]
        # Retain only channels 0 (IR1), 1 (WV), 2 (PMW)
        selected_channels = raw_matrix[:, :, :, :3]
        
        # Extract metadata
        c_ids = [s.decode("utf-8") if isinstance(s, bytes) else str(s) for s in f["info/cyclone_id"][:]]
        times = [s.decode("utf-8") if isinstance(s, bytes) else str(s) for s in f["info/time"][:]]
        vmax = f["info/vmax"][:]
        pres = f["info/pres"][:]
        lat = f["info/lat"][:]
        lon = f["info/lon"][:]
        
        meta_df = pd.DataFrame({
            "cyclone_id": c_ids,
            "timestamp": times,
            "vmax": vmax,
            "pres": pres,
            "lat": lat,
            "lon": lon
        })
        
    return selected_channels, meta_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TCIR-2017 Dataset Downloader & Sample Generator")
    parser.add_argument("--output", type=str, default="data/tcir_sample.h5", help="Path to save the HDF5 file")
    parser.add_argument("--generate-sample", action="store_true", default=True, help="Generate sample synthetic dataset")
    args = parser.parse_args()
    
    if args.generate_sample:
        create_sample_tcir_dataset(args.output)
