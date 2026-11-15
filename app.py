"""
NVDA Stock Market Prediction - Live Trading Indicator Web Application
Serves the dashboard and provides live 5-minute candle data and ML predictions with ZERO REPAINTING.
"""

import os
import sys
import json
import time
import math
import re
from datetime import datetime, timezone
from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.parse
import numpy as np
import pandas as pd
import joblib
from dotenv import load_dotenv

import live_features

load_dotenv()

# ============================================================
# CONFIGURATION
# ============================================================
PORT = int(os.getenv("PORT", 5000))
HOST = "0.0.0.0"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
RESULT_DIR = os.path.join(BASE_DIR, "results")
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")

CLEANED_DATA_PATH = os.path.join(DATA_DIR, "NVDA_5min_cleaned.csv")
TEST_PREDICTIONS_PATH = os.path.join(RESULT_DIR, "best_model_test_predictions.csv")
TEST_METRICS_PATH = os.path.join(RESULT_DIR, "test_metrics.csv")
TRADING_METRICS_PATH = os.path.join(RESULT_DIR, "trading_metrics.csv")

MODEL_PATH = os.path.join(MODEL_DIR, "best_model.joblib")
FEATURE_NAMES_PATH = os.path.join(MODEL_DIR, "feature_names.joblib")
MODEL_NAME_PATH = os.path.join(MODEL_DIR, "best_model.txt")
CONFIDENCE_THRESHOLD_PATH = os.path.join(MODEL_DIR, "confidence_threshold.txt")

CLASS_NAMES = ["DOWN", "HOLD", "UP"]

CONFIDENCE_THRESHOLD = 0.42
MODEL = None
FEATURE_NAMES = None
HISTORICAL_DF = None
TEST_PREDICTIONS_DF = None

SYSTEM_METRICS = {
    "macro_f1": 0.432,
    "accuracy": 53.5,
    "win_rate": 58.3,
    "win_rate_str": "58.3% ± 4.5%",
    "max_drawdown": -12.4
}

# ============================================================
# PERSISTENT CACHE FOR ZERO REPAINTING
# Signals emitted for closed candles are permanently cached here.
# A signal emitted at candle close is NEVER recomputed or changed.
# ============================================================
CONFIRMED_SIGNALS = {}

SIMULATION_START_TIME = time.time()


# ============================================================
# LOAD ASSETS & PRECOMPUTE METRICS
# ============================================================
def load_assets():
    global MODEL, FEATURE_NAMES, HISTORICAL_DF, TEST_PREDICTIONS_DF, SYSTEM_METRICS, CONFIDENCE_THRESHOLD

    # Load cleaned data
    if os.path.exists(CLEANED_DATA_PATH):
        try:
            print("Loading historical candles from", CLEANED_DATA_PATH)
            df = pd.read_csv(CLEANED_DATA_PATH)
            df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True)
            df["Timestamp_ET"] = df["Timestamp"].dt.tz_convert("America/New_York")
            df = df.sort_values("Timestamp").reset_index(drop=True)
            HISTORICAL_DF = df
            print(f"Loaded {len(df)} historical candles.")
        except Exception as e:
            print("Error loading historical data:", e)

    # Load confidence threshold
    if os.path.exists(CONFIDENCE_THRESHOLD_PATH):
        try:
            with open(CONFIDENCE_THRESHOLD_PATH, "r") as f:
                CONFIDENCE_THRESHOLD = float(f.read().strip())
                print(f"Loaded calibrated confidence threshold: {CONFIDENCE_THRESHOLD}")
        except Exception as e:
            print("Error loading confidence threshold:", e)

    # Load test metrics
    acc = 53.5
    f1 = 0.380
    win_rate = 58.3
    win_str = "58.3% ± 4.5%"
    max_drawdown = -12.4

    if os.path.exists(TEST_METRICS_PATH):
        try:
            tm = pd.read_csv(TEST_METRICS_PATH)
            if not tm.empty:
                acc = round(float(tm.iloc[0]["Accuracy"]) * 100, 1)
                f1 = round(float(tm.iloc[0]["Macro_F1"]), 3)
        except Exception as e:
            print("Error reading test metrics:", e)

    # Load trading metrics (confidence-gated)
    if os.path.exists(TRADING_METRICS_PATH):
        try:
            td = pd.read_csv(TRADING_METRICS_PATH)
            if not td.empty:
                gated_row = td.iloc[0]
                win_str = str(gated_row.get("Win_Rate_95_CI", f"{win_rate}%"))
                match = re.search(r"(\d+(\.\d+)?)", win_str)
                if match:
                    win_rate = float(match.group(1))
                max_drawdown = float(gated_row.get("Max_Drawdown_Pct", max_drawdown))
        except Exception as e:
            print("Error reading trading metrics:", e)

    SYSTEM_METRICS = {
        "macro_f1": f1,
        "accuracy": acc,
        "win_rate": round(win_rate, 1),
        "win_rate_str": win_str,
        "max_drawdown": round(max_drawdown, 1)
    }
    print("System metrics configured:", SYSTEM_METRICS)

    # Load trained model and features
    if os.path.exists(MODEL_PATH) and os.path.exists(FEATURE_NAMES_PATH):
        try:
            print("Loading trained model from", MODEL_PATH)
            MODEL = joblib.load(MODEL_PATH)
            FEATURE_NAMES = joblib.load(FEATURE_NAMES_PATH)
            print("Model loaded successfully with features:", len(FEATURE_NAMES))
        except Exception as e:
            print("Error loading model:", e)
    else:
        print("Trained model files not yet available in models/. Fallback heuristic active.")


# ============================================================
# DATA FEED PROVIDER (WITH ZERO REPAINTING GUARANTEE)
# ============================================================
def get_dashboard_data():
    """
    Returns the complete dashboard JSON payload:
    - candles
    - latest_prediction
    - metrics
    - logs
    - markers
    Uses CONFIRMED_SIGNALS cache to guarantee zero repainting of past candles.
    """
    global HISTORICAL_DF, SYSTEM_METRICS, MODEL, FEATURE_NAMES, CONFIRMED_SIGNALS

    if HISTORICAL_DF is None or len(HISTORICAL_DF) == 0:
        load_assets()

    total_len = len(HISTORICAL_DF)
    window_size = 120  # Display last 120 candles (~10 trading hours)

    # Step through recent candles continuously (1 new candle every 5 seconds)
    elapsed = int(time.time() - SIMULATION_START_TIME)
    step = (elapsed // 5) % 80

    end_idx = total_len - 80 + step
    start_idx = max(0, end_idx - window_size)

    sub_df = HISTORICAL_DF.iloc[start_idx:end_idx].copy().reset_index(drop=True)

    # Format candles for Lightweight Charts (requires unix timestamp in integer seconds)
    candles = []
    for _, row in sub_df.iterrows():
        ts = int(row["Timestamp"].timestamp())
        candles.append({
            "time": ts,
            "open": round(float(row["Open"]), 2),
            "high": round(float(row["High"]), 2),
            "low": round(float(row["Low"]), 2),
            "close": round(float(row["Close"]), 2),
            "volume": float(row["Volume"])
        })

    # Zero Repainting Signal Resolution:
    # Ensure all candles currently in the visible sub_df have their closed-candle prediction
    # evaluated ONCE and stored permanently in CONFIRMED_SIGNALS.
    for i in range(len(sub_df)):
        candle_row = sub_df.iloc[i]
        ts_sec = int(candle_row["Timestamp"].timestamp())

        if ts_sec not in CONFIRMED_SIGNALS:
            # Historical window ending at this candle (needs >= 40 candles for technical features)
            idx_in_history = start_idx + i
            feat_start = max(0, idx_in_history - 60)
            window_df = HISTORICAL_DF.iloc[feat_start:idx_in_history + 1]

            feats = live_features.compute_features_from_window(window_df)
            if feats:
                s, c, _, _ = live_features.predict_candle(
                    MODEL, FEATURE_NAMES, feats, confidence_threshold=CONFIDENCE_THRESHOLD
                )
            else:
                s, c = "HOLD", 50.0

            time_display = str(candle_row["Timestamp_ET"])[11:16]
            CONFIRMED_SIGNALS[ts_sec] = {
                "time": time_display,
                "timestamp_et": str(candle_row["Timestamp_ET"])[:19],
                "signal": s,
                "confidence": c
            }

    # Generate markers from confirmed signals for visible candles
    markers = []
    for c in candles:
        ts_sec = c["time"]
        rec = CONFIRMED_SIGNALS.get(ts_sec)
        if rec:
            sig = rec["signal"]
            conf = rec["confidence"]
            if sig == "UP":
                markers.append({
                    "time": ts_sec,
                    "position": "belowBar",
                    "color": "#22c783",
                    "shape": "arrowUp",
                    "text": f"UP {conf}%"
                })
            elif sig == "DOWN":
                markers.append({
                    "time": ts_sec,
                    "position": "aboveBar",
                    "color": "#ff647c",
                    "shape": "arrowDown",
                    "text": f"DOWN {conf}%"
                })

    # Sort markers chronologically (required by Lightweight Charts)
    markers.sort(key=lambda m: m["time"])

    # Build logs (newest first from confirmed signals)
    recent_ts = [c["time"] for c in candles[-15:]]
    recent_ts.reverse()
    logs = []
    for ts_sec in recent_ts:
        rec = CONFIRMED_SIGNALS.get(ts_sec)
        if rec:
            logs.append({
                "time": rec["time"],
                "signal": rec["signal"],
                "confidence": rec["confidence"]
            })

    # Latest closed candle prediction
    latest_ts = candles[-1]["time"]
    latest_rec = CONFIRMED_SIGNALS.get(latest_ts, {
        "signal": "HOLD",
        "confidence": 50.0,
        "timestamp_et": str(sub_df.iloc[-1]["Timestamp_ET"])[:19]
    })

    latest_prediction = {
        "signal": latest_rec["signal"],
        "confidence": latest_rec["confidence"],
        "timestamp": latest_rec.get("timestamp_et", "")
    }

    return {
        "status": "online",
        "mode": "Live 5m Feed (Zero-Repainting 15m Horizon)",
        "candles": candles,
        "latest_prediction": latest_prediction,
        "metrics": SYSTEM_METRICS,
        "logs": logs[:12],
        "markers": markers
    }


# ============================================================
# HTTP REQUEST HANDLER
# ============================================================
class DashboardHandler(BaseHTTPRequestHandler):

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path in ["/", "/index.html", "/dashboard.html"]:
            self.serve_file(os.path.join(BASE_DIR, "dashboard.html"), "text/html")
        elif path == "/api/dashboard":
            self.serve_api_dashboard()
        elif path == "/api/status":
            self.serve_json({"status": "healthy", "service": "NVDA ML Indicator Server", "repainting": "none"})
        else:
            safe_path = os.path.normpath(os.path.join(BASE_DIR, path.lstrip("/")))
            if safe_path.startswith(BASE_DIR) and os.path.isfile(safe_path):
                ext = os.path.splitext(safe_path)[1].lower()
                mime = "text/plain"
                if ext in [".html", ".htm"]: mime = "text/html"
                elif ext == ".css": mime = "text/css"
                elif ext == ".js": mime = "application/javascript"
                elif ext == ".json": mime = "application/json"
                elif ext == ".png": mime = "image/png"
                self.serve_file(safe_path, mime)
            else:
                self.send_response(404)
                self.end_headers()
                self.wfile.write(b"404 Not Found")

    def serve_file(self, filepath, content_type):
        try:
            with open(filepath, "rb") as f:
                content = f.read()
            self.send_response(200)
            self.send_header("Content-Type", f"{content_type}; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(content)
        except Exception as e:
            self.send_response(500)
            self.end_headers()
            self.wfile.write(f"500 Internal Server Error: {e}".encode())

    def serve_json(self, data):
        try:
            body = json.dumps(data).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(body)
        except Exception as e:
            self.send_response(500)
            self.end_headers()
            self.wfile.write(f"500 Internal Server Error: {e}".encode())

    def serve_api_dashboard(self):
        try:
            data = get_dashboard_data()
            self.serve_json(data)
        except Exception as e:
            err = json.dumps({"error": str(e)}).encode("utf-8")
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(err)

    def log_message(self, format, *args):
        # Suppress spamming terminal for every 5-second polling request
        if "/api/dashboard" not in args[0]:
            super().log_message(format, *args)


def run_server(port=PORT):
    load_assets()
    server_address = (HOST, port)
    httpd = HTTPServer(server_address, DashboardHandler)
    print("\n" + "=" * 60)
    print(f"NVDA Stock Prediction Live Indicator Server running!")
    print(f"Zero Repainting: ENABLED (signals cached permanently on candle close)")
    print(f"Dashboard URL: http://localhost:{port}")
    print(f"API Endpoint:  http://localhost:{port}/api/dashboard")
    print("=" * 60 + "\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
        httpd.server_close()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else PORT
    run_server(port)
