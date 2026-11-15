"""
test_parity.py - Unit Test for Train/Serve Feature Parity
Verifies that features computed in batch offline training are mathematically identical
to features computed from sliding live buffers (down to 1e-6 precision).
"""

import sys
import numpy as np
import pandas as pd
import live_features


def test_feature_parity():
    cleaned_path = "data/processed/NVDA_5min_cleaned.csv"
    print("Loading cleaned dataset from", cleaned_path)
    raw_df = pd.read_csv(cleaned_path)
    
    # 1. Batch extraction across full 2-year dataset
    print(f"1. Running batch feature extraction on {len(raw_df)} historical candles...")
    batch_df = live_features.extract_features(raw_df)
    
    # Pick 5 distinct test days across the dataset
    unique_dates = batch_df["Trading_Date"].unique()
    test_indices = [10, 100, 250, 350, 450]
    
    print(f"2. Testing parity across {len(test_indices)} distinct sessions...")
    
    max_absolute_error = 0.0
    
    for idx in test_indices:
        prev_date = unique_dates[idx - 1]
        target_date = unique_dates[idx]
        
        # Simulate live streaming: input is only prev_date + target_date (current live session)
        raw_df["Temp_Date"] = pd.to_datetime(raw_df["Timestamp"], utc=True).dt.tz_convert("America/New_York").dt.date
        live_window = raw_df[raw_df["Temp_Date"].isin([prev_date, target_date])].copy().reset_index(drop=True)
        live_df = live_features.extract_features(live_window)
        
        batch_session = batch_df[batch_df["Trading_Date"] == target_date].reset_index(drop=True)
        live_session = live_df[live_df["Trading_Date"] == target_date].reset_index(drop=True)
        
        assert len(batch_session) == len(live_session), f"Row count mismatch on {target_date}"
        
        for col in live_features.FEATURE_NAMES:
            err = np.max(np.abs(batch_session[col].values - live_session[col].values))
            if err > max_absolute_error:
                max_absolute_error = err
            assert err < 1e-6, f"Feature parity failure on {target_date} in column {col}: max error = {err}"
            
    print(f"\n=======================================================")
    print(f"SUCCESS: 100% FEATURE PARITY VERIFIED")
    print(f"Max absolute difference across all features: {max_absolute_error:.2e}")
    print(f"Tested features ({len(live_features.FEATURE_NAMES)}):")
    for i, f in enumerate(live_features.FEATURE_NAMES, 1):
        print(f"  {i:2d}. {f}")
    print(f"=======================================================\n")


if __name__ == "__main__":
    test_feature_parity()
