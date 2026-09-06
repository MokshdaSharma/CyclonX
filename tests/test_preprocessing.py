"""
CycloneX - Preprocessing & Partitioning Unit Tests
Validates robust channel normalization, boundary scaling, and cyclone-level partition disjointness.
"""

import os
import sys
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "configs"))

from preprocess import (
    cyclone_level_train_val_test_split,
    apply_robust_normalization,
    compute_robust_channel_stats,
    get_intensity_category
)


def test_intensity_categorization():
    assert get_intensity_category(25.0) == "Weak"
    assert get_intensity_category(33.9) == "Weak"
    assert get_intensity_category(34.0) == "Moderate"
    assert get_intensity_category(63.5) == "Moderate"
    assert get_intensity_category(64.0) == "Strong"
    assert get_intensity_category(82.9) == "Strong"
    assert get_intensity_category(83.0) == "Very Strong"
    assert get_intensity_category(135.0) == "Very Strong"


def test_cyclone_level_split_has_zero_leakage():
    cyclone_ids = []
    for c in ["STORM_A", "STORM_B", "STORM_C", "STORM_D", "STORM_E", "STORM_F", "STORM_G", "STORM_H", "STORM_I", "STORM_J"]:
        for _ in range(6):
            cyclone_ids.append(c)
            
    df = pd.DataFrame({
        "cyclone_id": cyclone_ids,
        "vmax": np.random.uniform(30, 100, len(cyclone_ids))
    })
    
    train_idx, val_idx, test_idx = cyclone_level_train_val_test_split(df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
    
    train_storms = set(df.iloc[train_idx]["cyclone_id"])
    val_storms = set(df.iloc[val_idx]["cyclone_id"])
    test_storms = set(df.iloc[test_idx]["cyclone_id"])
    
    # Assert zero storm identity leakage
    assert len(train_storms.intersection(val_storms)) == 0, "Train and Val share cyclones!"
    assert len(train_storms.intersection(test_storms)) == 0, "Train and Test share cyclones!"
    assert len(val_storms.intersection(test_storms)) == 0, "Val and Test share cyclones!"
    assert len(train_idx) + len(val_idx) + len(test_idx) == len(df)


def test_robust_normalization_scaling():
    # Shape (10, 201, 201, 3)
    np.random.seed(42)
    fake_images = np.random.normal(loc=260.0, scale=15.0, size=(10, 201, 201, 3)).astype(np.float32)
    
    stats = compute_robust_channel_stats(fake_images[:6], config_output_path="configs/test_preprocessing_config.json")
    assert "IR1" in stats and "WV" in stats and "PMW" in stats
    
    norm_imgs = apply_robust_normalization(fake_images, stats)
    
    # Values should be clipped and scaled within [0.0, 255.0]
    assert norm_imgs.shape == fake_images.shape
    assert norm_imgs.min() >= 0.0, f"Min was {norm_imgs.min()}"
    assert norm_imgs.max() <= 255.0, f"Max was {norm_imgs.max()}"
    
    # Cleanup test config
    if os.path.exists("configs/test_preprocessing_config.json"):
        os.remove("configs/test_preprocessing_config.json")
