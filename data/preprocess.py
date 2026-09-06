"""
CycloneX - Data Preprocessing & Cyclone-Level Partitioning
Performs robust normalization on satellite imagery (IR1, WV, PMW),
computes normalization parameters strictly on the training split to avoid data leakage,
and enforces cyclone-level splitting (train/val/test ~70/15/15).

DISCLAIMER: AI-assisted research prototype for decision support. Not an official IMD warning.
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, List


CHANNEL_NAMES = ["IR1", "WV", "PMW"]


def get_intensity_category(vmax: float) -> str:
    """
    Maps continuous Vmax (knots) to standard 4 intensity categories:
    - Weak: < 34 kt
    - Moderate: 34 - 63 kt
    - Strong: 64 - 82 kt
    - Very Strong: >= 83 kt
    """
    if vmax < 34.0:
        return "Weak"
    elif vmax < 64.0:
        return "Moderate"
    elif vmax < 83.0:
        return "Strong"
    else:
        return "Very Strong"


def cyclone_level_train_val_test_split(
    meta_df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Partitions dataset by unique cyclone IDs rather than individual frames.
    Guarantees ZERO overlap of cyclones between train, val, and test splits.
    """
    unique_cyclones = meta_df["cyclone_id"].unique()
    np.random.seed(random_state)
    shuffled_cyclones = np.random.permutation(unique_cyclones)
    
    n_total = len(shuffled_cyclones)
    n_train = max(1, int(round(n_total * train_ratio)))
    n_val = max(1, int(round(n_total * val_ratio)))
    
    # Ensure remaining goes to test
    train_cyclones = set(shuffled_cyclones[:n_train])
    val_cyclones = set(shuffled_cyclones[n_train:n_train + n_val])
    test_cyclones = set(shuffled_cyclones[n_train + n_val:])
    
    if not test_cyclones and len(val_cyclones) > 1:
        # If total cyclone count is small, shift 1 from val to test
        transferred = val_cyclones.pop()
        test_cyclones.add(transferred)
    elif not test_cyclones:
        # Fallback for very small sample: make copy
        test_cyclones = set(shuffled_cyclones[-1:])
        
    train_indices = meta_df[meta_df["cyclone_id"].isin(train_cyclones)].index.values
    val_indices = meta_df[meta_df["cyclone_id"].isin(val_cyclones)].index.values
    test_indices = meta_df[meta_df["cyclone_id"].isin(test_cyclones)].index.values
    
    # Sanity checks
    assert len(train_cyclones.intersection(val_cyclones)) == 0, "Train and Val share cyclones!"
    assert len(train_cyclones.intersection(test_cyclones)) == 0, "Train and Test share cyclones!"
    assert len(val_cyclones.intersection(test_cyclones)) == 0, "Val and Test share cyclones!"
    
    print(f"Cyclone-Level Split: {len(train_cyclones)} Train, {len(val_cyclones)} Val, {len(test_cyclones)} Test cyclones.")
    print(f"Frame Counts: Train={len(train_indices)}, Val={len(val_indices)}, Test={len(test_indices)}")
    
    return train_indices, val_indices, test_indices


def compute_robust_channel_stats(
    train_images: np.ndarray,
    config_output_path: str = "configs/preprocessing_config.json"
) -> Dict[str, Dict[str, float]]:
    """
    Computes robust normalization statistics (median, IQR scale)
    strictly from the training partition.
    Formula:
        scale = IQR / 1.349 (or 75th - 25th percentile)
    """
    stats = {}
    for i, ch_name in enumerate(CHANNEL_NAMES):
        channel_vals = train_images[:, :, :, i].flatten()
        # Filter non-finite if any
        valid_vals = channel_vals[np.isfinite(channel_vals)]
        if len(valid_vals) == 0:
            median_val = 273.15
            scale_val = 20.0
        else:
            median_val = float(np.median(valid_vals))
            q25 = float(np.percentile(valid_vals, 25))
            q75 = float(np.percentile(valid_vals, 75))
            iqr = q75 - q25
            scale_val = float(iqr / 1.349) if iqr > 1e-4 else 1.0
            
        stats[ch_name] = {
            "median": round(median_val, 4),
            "scale": round(scale_val, 4),
            "description": f"{ch_name} robust channel statistics (Kelvin)"
        }
        
    # Save to config file
    config_dict = {
        "channels": CHANNEL_NAMES,
        "image_shape": [201, 201, 3],
        "dropped_channels": ["VIS"],
        "normalization_method": "robust_scaling",
        "clip_range": [-2.0, 3.0],
        "target_range": [0.0, 255.0],
        "channel_stats": stats,
        "split_ratio": {"train": 0.70, "val": 0.15, "test": 0.15},
        "split_by": "cyclone_id",
        "disclaimer": "AI-assisted decision-support research prototype. Not an official IMD warning."
    }
    
    os.makedirs(os.path.dirname(config_output_path), exist_ok=True)
    with open(config_output_path, "w") as f:
        json.dump(config_dict, f, indent=2)
        
    print(f"Computed & saved robust preprocessing stats to: {config_output_path}")
    return stats


def apply_robust_normalization(
    images: np.ndarray,
    channel_stats: Dict[str, Dict[str, float]],
    clip_min: float = -2.0,
    clip_max: float = 3.0
) -> np.ndarray:
    """
    Normalizes images using precomputed channel statistics:
    1. z = (x - median) / scale
    2. z_clipped = clip(z, clip_min, clip_max)
    3. z_scaled = (z_clipped - clip_min) / (clip_max - clip_min) * 255.0
    Returns array in shape (N, 201, 201, 3) with range [0.0, 255.0] (float32).
    """
    norm_images = np.zeros_like(images, dtype=np.float32)
    
    for i, ch_name in enumerate(CHANNEL_NAMES):
        stat = channel_stats.get(ch_name, {"median": 273.15, "scale": 20.0})
        median = stat["median"]
        scale = max(stat["scale"], 1e-4)
        
        ch_data = images[:, :, :, i]
        # Replace NaNs with median
        ch_data = np.nan_to_num(ch_data, nan=median)
        
        # Robust z-score
        z = (ch_data - median) / scale
        
        # Clip to [-2, 3]
        z_clipped = np.clip(z, clip_min, clip_max)
        
        # Rescale to [0, 1] then multiply by 255
        z_rescaled = (z_clipped - clip_min) / (clip_max - clip_min) * 255.0
        norm_images[:, :, :, i] = z_rescaled
        
    return norm_images


def preprocess_dataset(
    raw_images: np.ndarray,
    meta_df: pd.DataFrame,
    config_path: str = "configs/preprocessing_config.json"
) -> Dict[str, Any]:
    """
    Full preprocessing pipeline:
    1. Cyclone-level train/val/test splitting
    2. Robust stat computation on train set
    3. Normalization of all splits
    4. Categorical label assignment
    """
    train_idx, val_idx, test_idx = cyclone_level_train_val_test_split(meta_df)
    
    # Compute stats strictly on train split
    stats = compute_robust_channel_stats(raw_images[train_idx], config_path)
    
    # Normalize images
    norm_images = apply_robust_normalization(raw_images, stats)
    
    # Add intensity categories to metadata
    meta_df = meta_df.copy()
    meta_df["category"] = meta_df["vmax"].apply(get_intensity_category)
    
    return {
        "images": norm_images,
        "metadata": meta_df,
        "splits": {
            "train_indices": train_idx,
            "val_indices": val_idx,
            "test_indices": test_idx
        },
        "stats": stats
    }


if __name__ == "__main__":
    from download_tcir import load_raw_tcir, create_sample_tcir_dataset
    sample_file = "data/tcir_sample.h5"
    if not os.path.exists(sample_file):
        create_sample_tcir_dataset(sample_file)
        
    images, meta = load_raw_tcir(sample_file)
    result = preprocess_dataset(images, meta)
    print("Preprocessing completed successfully!")
    print(f"Images shape: {result['images'].shape}, min: {result['images'].min():.2f}, max: {result['images'].max():.2f}")
