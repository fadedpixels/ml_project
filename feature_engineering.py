"""
feature_engineering.py - Offline Batch Feature Extraction & Dataset Preparation
Uses live_features module to ensure 100% parity with live streaming.
"""

import os
import pandas as pd
import live_features

INPUT_FILE = "data/processed/NVDA_5min_cleaned.csv"
OUTPUT_DIR = "data/processed"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "NVDA_features.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)


def run_feature_engineering():
    print("\n" + "=" * 55)
    print("RUNNING FEATURE ENGINEERING (PARITY-ALIGNED)")
    print("=" * 55)
    
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")
        
    print(f"Loading cleaned data from {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)
    print("Initial candles:", len(df))
    
    # Extract all 28 stationary features using shared live_features module
    print("\nExtracting features via live_features.extract_features...")
    df_features = live_features.extract_features(df)
    
    # Create targets with dynamic volatility-adaptive thresholds
    print("Generating 15-minute prediction targets...")
    targets, fut_ret, fut_ts, valid = live_features.create_targets(df_features, prediction_candles=3, dynamic=True)
    
    df_features["Target"] = targets
    df_features["Future_Return"] = fut_ret
    df_features["Target_End_Timestamp"] = fut_ts
    
    # Keep only valid market session targets (drops last 3 candles of each day)
    clean_df = df_features[valid].copy().reset_index(drop=True)
    clean_df["Target"] = clean_df["Target"].astype(int)
    
    print("\nFinal feature dataset shape:", clean_df.shape)
    print("Target distribution:")
    target_counts = clean_df["Target"].value_counts().sort_index()
    for label, count in target_counts.items():
        name = live_features.CLASS_NAMES[label]
        pct = count / len(clean_df) * 100
        print(f"  Class {label} ({name:4s}): {count:6d} ({pct:5.1f}%)")
        
    # Verify no missing values in features
    null_counts = clean_df[live_features.FEATURE_NAMES].isnull().sum()
    max_nulls = null_counts.max()
    print(f"\nMissing values in feature columns: {max_nulls}")
    
    clean_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved feature dataset successfully to: {OUTPUT_FILE}")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    run_feature_engineering()
