"""
main.py - Main Entry Point for NVDA Stock Prediction & Live Indicator
Imports live_features and launches either the live dashboard or backtest verification.
"""

import sys
import os
import argparse
import pandas as pd
import joblib

# Import unified feature engineering & inference module
import live_features
import app


def print_evaluation_summary():
    """Prints the latest validation and test evaluation metrics."""
    res_dir = os.path.join(os.path.dirname(__file__), "results")
    test_metrics_path = os.path.join(res_dir, "test_metrics.csv")
    trading_metrics_path = os.path.join(res_dir, "trading_metrics.csv")

    print("\n" + "=" * 65)
    print("NVDA 15-MINUTE PREDICTION - MODEL EVALUATION SUMMARY")
    print("=" * 65)

    if os.path.exists(test_metrics_path):
        print("\n--- STATISTICAL CLASSIFICATION METRICS (UNTOUCHED TEST SET) ---")
        df_test = pd.read_csv(test_metrics_path)
        print(df_test.to_string(index=False))

    if os.path.exists(trading_metrics_path):
        print("\n--- SIMULATED TRADING PERFORMANCE (WITH TRANSACTION COSTS) ---")
        df_trade = pd.read_csv(trading_metrics_path)
        print(df_trade.to_string(index=False))

    print("\n" + "=" * 65)


def run_live_verification():
    """Performs a live feature extraction and inference check on the latest candle."""
    cleaned_path = os.path.join(os.path.dirname(__file__), "data", "processed", "NVDA_5min_cleaned.csv")
    model_path = os.path.join(os.path.dirname(__file__), "models", "best_model.joblib")
    features_path = os.path.join(os.path.dirname(__file__), "models", "feature_names.joblib")

    if not os.path.exists(cleaned_path):
        print("Data file not found at", cleaned_path)
        return

    print("\nRunning live feature extraction and inference verification...")
    df = pd.read_csv(cleaned_path).tail(60)
    feats = live_features.compute_features_from_window(df)
    print(f"Extracted {len(feats)} live stationary features successfully.")

    if os.path.exists(model_path) and os.path.exists(features_path):
        model = joblib.load(model_path)
        feat_names = joblib.load(features_path)
        signal, conf, prob_dict, is_conf = live_features.predict_candle(model, feat_names, feats)
        print(f"Latest Live Signal: {signal} ({conf}%) | Probabilities: {prob_dict} | Confident Gate: {is_conf}")
    else:
        print("Trained model not yet found. Please run: python train_models.py")


def main():
    parser = argparse.ArgumentParser(description="NVDA Intraday AI Trading Indicator")
    parser.add_argument("--eval", action="store_true", help="Print model and trading evaluation summary")
    parser.add_argument("--test", action="store_true", help="Run live feature extraction test")
    parser.add_argument("--port", type=int, default=5000, help="Web dashboard port (default: 5000)")
    args = parser.parse_args()

    if args.eval:
        print_evaluation_summary()
    elif args.test:
        run_live_verification()
    else:
        # Default behavior: launch the live dashboard web server
        print_evaluation_summary()
        app.run_server(port=args.port)


if __name__ == "__main__":
    main()
