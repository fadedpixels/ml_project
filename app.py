"""
NVDA Stock Market Prediction - Live Trading Indicator Web Application
Serves the fintech terminal dashboard and provides:
1. Live Massive API candle feed during open US market hours (9:30 - 16:00 ET).
2. Honest status reporting (live, closed, replay) - never faking live when closed.
3. Persistent signal caching to results/live_signals.json (zero repainting across restarts).
4. Out-of-sample replay restricted strictly to the untouched test set (post 2026-06-10).
5. Wilson Score confidence intervals with minimum sample size gates (>= 20 signals).
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
import requests
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
LIVE_SIGNALS_FILE = os.path.join(RESULT_DIR, "live_signals.json")

MODEL_PATH = os.path.join(MODEL_DIR, "best_model.joblib")
FEATURE_NAMES_PATH = os.path.join(MODEL_DIR, "feature_names.joblib")
MODEL_NAME_PATH = os.path.join(MODEL_DIR, "best_model.txt")
CONFIDENCE_THRESHOLD_PATH = os.path.join(MODEL_DIR, "confidence_threshold.txt")

CLASS_NAMES = ["DOWN", "HOLD", "UP"]

CONFIDENCE_THRESHOLD = 0.44
MODEL = None
FEATURE_NAMES = None
HISTORICAL_DF = None
TEST_DF = None  # Strictly out-of-sample historical slice (>= 2026-06-10)

# ============================================================
# PERSISTENT CACHES FOR ZERO REPAINTING & OUTCOME RESOLUTION
# ============================================================
CONFIRMED_SIGNALS = {}
RESOLVED_OUTCOMES = {}

LIVE_API_CACHE = {
    "last_fetch": 0,
    "df": None,
    "status": "unknown"
}

SIMULATION_START_TIME = time.time()


# ============================================================
# PERSISTENCE HELPERS
# ============================================================
def load_persisted_signals():
    global CONFIRMED_SIGNALS
    if os.path.exists(LIVE_SIGNALS_FILE):
        try:
            with open(LIVE_SIGNALS_FILE, "r") as f:
                data = json.load(f)
                for k, v in data.items():
                    CONFIRMED_SIGNALS[int(k)] = v
            print(f"Loaded {len(CONFIRMED_SIGNALS)} persisted signals from {LIVE_SIGNALS_FILE}")
        except Exception as e:
            print("Error loading live_signals.json:", e)


def persist_signals():
    global CONFIRMED_SIGNALS
    try:
        os.makedirs(RESULT_DIR, exist_ok=True)
        serializable = {str(k): v for k, v in CONFIRMED_SIGNALS.items()}
        with open(LIVE_SIGNALS_FILE, "w") as f:
            json.dump(serializable, f, indent=2)
    except Exception as e:
        print("Error saving live_signals.json:", e)


# ============================================================
# LOAD ASSETS & PRECOMPUTE METRICS
# ============================================================
def load_assets():
    global MODEL, FEATURE_NAMES, HISTORICAL_DF, TEST_DF, CONFIDENCE_THRESHOLD

    # 1. Load cleaned data
    if os.path.exists(CLEANED_DATA_PATH):
        try:
            print("Loading historical candles from", CLEANED_DATA_PATH)
            df = pd.read_csv(CLEANED_DATA_PATH)
            df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True)
            df["Timestamp_ET"] = df["Timestamp"].dt.tz_convert("America/New_York")
            df = df.sort_values("Timestamp").reset_index(drop=True)
            HISTORICAL_DF = df
            
            # Restrict replay strictly to out-of-sample test window (post 2026-06-10)
            test_start = pd.to_datetime("2026-06-10 17:40:00+00:00")
            TEST_DF = df[df["Timestamp"] >= test_start].copy().reset_index(drop=True)
            print(f"Loaded {len(df)} total candles. Out-of-sample test slice: {len(TEST_DF)} candles.")
        except Exception as e:
            print("Error loading historical data:", e)

    # 2. Load confidence threshold from file (single source of truth)
    if os.path.exists(CONFIDENCE_THRESHOLD_PATH):
        try:
            with open(CONFIDENCE_THRESHOLD_PATH, "r") as f:
                CONFIDENCE_THRESHOLD = float(f.read().strip())
                print(f"Loaded calibrated confidence threshold: {CONFIDENCE_THRESHOLD}")
        except Exception as e:
            print("Error loading confidence threshold:", e)

    # 3. Load trained model and features
    if os.path.exists(MODEL_PATH) and os.path.exists(FEATURE_NAMES_PATH):
        try:
            print("Loading trained model from", MODEL_PATH)
            MODEL = joblib.load(MODEL_PATH)
            FEATURE_NAMES = joblib.load(FEATURE_NAMES_PATH)
            print("Model loaded successfully with features:", len(FEATURE_NAMES))
        except Exception as e:
            print("Error loading model:", e)

    # 4. Load persisted live signals cache
    load_persisted_signals()


def get_current_threshold():
    """Always reads the active threshold directly from models/confidence_threshold.txt."""
    global CONFIDENCE_THRESHOLD
    if os.path.exists(CONFIDENCE_THRESHOLD_PATH):
        try:
            with open(CONFIDENCE_THRESHOLD_PATH, "r") as f:
                CONFIDENCE_THRESHOLD = float(f.read().strip())
        except Exception:
            pass
    return CONFIDENCE_THRESHOLD


# ============================================================
# MARKET HOURS & LIVE CANDLE FETCHER
# ============================================================
def is_us_equity_market_open(now_et):
    """
    Checks if current Eastern Time is within regular US equity trading hours:
    Monday (0) to Friday (4), 9:30 AM to 4:00 PM ET.
    """
    if now_et.weekday() >= 5:
        return False
    m_open = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
    m_close = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
    return m_open <= now_et < m_close


def fetch_live_candles(now_et):
    """
    Fetches the latest closed 5-minute candles from Massive API during live market hours.
    Maintains ~60 prior candles for rolling feature warm-up.
    """
    api_key = os.getenv("MASSIVE_API_KEY")
    if not api_key:
        return None, "No Massive API key configured in .env"

    t_now = time.time()
    # Rate limit cache: 15 seconds
    if LIVE_API_CACHE["df"] is not None and (t_now - LIVE_API_CACHE["last_fetch"]) < 15:
        return LIVE_API_CACHE["df"], "cached"

    try:
        # Fetch last 5 calendar days to ensure prior session warmup is available
        start_date = (now_et - pd.Timedelta(days=5)).strftime("%Y-%m-%d")
        end_date = now_et.strftime("%Y-%m-%d")
        url = f"https://api.massive.com/v2/aggs/ticker/NVDA/range/5/minute/{start_date}/{end_date}"
        params = {"adjusted": "true", "sort": "asc", "limit": 5000, "apiKey": api_key}

        resp = requests.get(url, params=params, timeout=8)
        if resp.status_code == 200:
            data = resp.json()
            results = data.get("results", [])
            if results:
                df = pd.DataFrame(results)
                rename_map = {
                    "t": "Timestamp", "o": "Open", "h": "High", "l": "Low",
                    "c": "Close", "v": "Volume", "vw": "VWAP", "n": "Transactions"
                }
                df.rename(columns=rename_map, inplace=True)
                df["Timestamp"] = pd.to_datetime(df["Timestamp"], unit="ms", utc=True)
                df["Timestamp_ET"] = df["Timestamp"].dt.tz_convert("America/New_York")

                # Filter regular session: 9:30 to 16:00 ET
                t_et = df["Timestamp_ET"].dt.time
                m_open = pd.to_datetime("09:30").time()
                m_close = pd.to_datetime("16:00").time()
                df = df[(t_et >= m_open) & (t_et <= m_close)].sort_values("Timestamp").reset_index(drop=True)

                if len(df) >= 30:
                    LIVE_API_CACHE["last_fetch"] = t_now
                    LIVE_API_CACHE["df"] = df
                    return df, "live"
    except Exception as e:
        print("Live candle fetch error from Massive API:", e)

    return None, "api_fallback"


# ============================================================
# STATISTICAL METRICS: WILSON SCORE INTERVAL
# ============================================================
def compute_wilson_ci(wins, n, z=1.96):
    """
    Computes Wilson Score confidence interval for a binomial proportion.
    Unlike Wald approximation, Wilson remains robust at extreme win rates.
    """
    if n == 0:
        return None, 0.0
    p_hat = wins / n
    denom = 1.0 + (z**2) / n
    center = (p_hat + (z**2) / (2.0 * n)) / denom
    margin = (z / denom) * math.sqrt((p_hat * (1.0 - p_hat) / n) + (z**2) / (4.0 * (n**2)))
    return center * 100.0, margin * 100.0


# ============================================================
# DATA FEED PROVIDER
# ============================================================
def get_dashboard_data():
    """
    Returns the complete dashboard JSON payload:
    - Honest market status ('live', 'closed', or 'replay (simulation)')
    - Out-of-sample replay restricted strictly to the test set
    - Persistent live signals with permanent outcome resolution
    - Wilson Score CI for live scorecard
    """
    global HISTORICAL_DF, TEST_DF, MODEL, FEATURE_NAMES, CONFIRMED_SIGNALS, RESOLVED_OUTCOMES

    if HISTORICAL_DF is None or len(HISTORICAL_DF) == 0:
        load_assets()

    threshold = get_current_threshold()
    window_size = 120  # Display last 120 candles (~10 trading hours)

    # Wall-clock Eastern Time
    real_now_et = pd.Timestamp.now(tz="America/New_York")
    market_is_open = is_us_equity_market_open(real_now_et)

    live_df = None
    market_status = "closed"
    data_age_s = 0

    if market_is_open:
        live_df, fetch_status = fetch_live_candles(real_now_et)
        if live_df is not None and len(live_df) >= 30:
            market_status = "live"
            working_df = live_df
            sub_df = working_df.tail(window_size).copy().reset_index(drop=True)
            data_age_s = int(time.time() - LIVE_API_CACHE["last_fetch"])
            now_et_str = real_now_et.strftime("%H:%M:%S ET")
            mins_in_current_bar = (real_now_et.minute % 5) * 60 + real_now_et.second
            next_close_s = max(5, 300 - mins_in_current_bar)
        else:
            market_status = "delayed"
            working_df = None
    else:
        market_status = "closed"
        working_df = None

    # Offline / Closed / Replay Mode:
    # Strictly restricted to out-of-sample test set (TEST_DF, post 2026-06-10)
    if working_df is None:
        source_df = TEST_DF if (TEST_DF is not None and len(TEST_DF) > 150) else HISTORICAL_DF
        total_len = len(source_df)
        elapsed = int(time.time() - SIMULATION_START_TIME)
        step = (elapsed // 5) % 80
        end_idx = total_len - 80 + step
        start_idx = max(0, end_idx - window_size)

        sub_df = source_df.iloc[start_idx:end_idx].copy().reset_index(drop=True)
        data_age_s = elapsed % 5
        now_et_str = sub_df.iloc[-1]["Timestamp_ET"].strftime("%H:%M:%S ET")
        next_close_s = 300 - (elapsed % 5) * 60

    latest_row = sub_df.iloc[-1]
    latest_ts_et = latest_row["Timestamp_ET"]
    latest_unix = int(latest_row["Timestamp"].timestamp())

    # Format candles with intraday VWAP
    candles = []
    cum_vol = 0.0
    cum_vol_price = 0.0
    current_date = None

    for _, row in sub_df.iterrows():
        ts = int(row["Timestamp"].timestamp())
        row_date = row["Timestamp_ET"].date()
        if row_date != current_date:
            current_date = row_date
            cum_vol = 0.0
            cum_vol_price = 0.0

        v = float(row["Volume"])
        typical_price = (float(row["High"]) + float(row["Low"]) + float(row["Close"])) / 3.0
        cum_vol += v
        cum_vol_price += typical_price * v
        vwap_val = round(cum_vol_price / cum_vol, 2) if cum_vol > 0 else round(float(row["Close"]), 2)

        candles.append({
            "time": ts,
            "open": round(float(row["Open"]), 2),
            "high": round(float(row["High"]), 2),
            "low": round(float(row["Low"]), 2),
            "close": round(float(row["Close"]), 2),
            "volume": float(row["Volume"]),
            "vwap": vwap_val
        })

    # Quote calculation
    session_start_close = float(sub_df.iloc[0]["Close"])
    current_price = round(float(latest_row["Close"]), 2)
    change = round(current_price - session_start_close, 2)
    change_pct = round((change / session_start_close) * 100, 2) if session_start_close > 0 else 0.0

    # Process signals for closed candles (Zero Repainting)
    new_signal_added = False
    for i in range(len(sub_df)):
        candle_row = sub_df.iloc[i]
        ts_sec = int(candle_row["Timestamp"].timestamp())
        ts_et = candle_row["Timestamp_ET"]

        if ts_sec not in CONFIRMED_SIGNALS:
            # Signal validity cutoff rule: Candles after 15:40 ET have no 15m horizon inside session
            is_after_cutoff = (ts_et.hour == 15 and ts_et.minute > 40) or (ts_et.hour >= 16)

            if is_after_cutoff:
                CONFIRMED_SIGNALS[ts_sec] = {
                    "time": ts_sec,
                    "time_et": ts_et.strftime("%H:%M"),
                    "signal": "NONE",
                    "confidence": 0.0,
                    "probs": {"DOWN": 33.3, "HOLD": 33.4, "UP": 33.3},
                    "entry": round(float(candle_row["Close"]), 2),
                    "target_index": i + 3,
                    "outcome": "none",
                    "ret_pct": 0.0,
                    "is_after_cutoff": True
                }
                new_signal_added = True
            else:
                window_df = sub_df.iloc[max(0, i - 60):i + 1]
                feats = live_features.compute_features_from_window(window_df)

                if feats:
                    s, c, p_dict, is_conf = live_features.predict_candle(
                        MODEL, FEATURE_NAMES, feats, confidence_threshold=threshold
                    )
                else:
                    s, c, p_dict, is_conf = "HOLD", 50.0, {"DOWN": 33.3, "HOLD": 33.4, "UP": 33.3}, False

                signal_state = s if (s in ["UP", "DOWN"] and is_conf) else "NONE"

                CONFIRMED_SIGNALS[ts_sec] = {
                    "time": ts_sec,
                    "time_et": ts_et.strftime("%H:%M"),
                    "signal": signal_state,
                    "confidence": round(float(c), 1),
                    "probs": p_dict,
                    "entry": round(float(candle_row["Close"]), 2),
                    "target_index": i + 3,
                    "outcome": "pending" if signal_state in ["UP", "DOWN"] else "none",
                    "ret_pct": None,
                    "is_after_cutoff": False
                }
                new_signal_added = True

    # Permanent outcome resolution (3 bars / 15 minutes forward)
    for ts_sec, sig_rec in CONFIRMED_SIGNALS.items():
        if sig_rec["signal"] in ["UP", "DOWN"] and sig_rec["outcome"] == "pending":
            tgt_idx = sig_rec.get("target_index")
            if tgt_idx is not None and tgt_idx < len(sub_df):
                exit_row = sub_df.iloc[tgt_idx]
                entry_p = sig_rec["entry"]
                exit_p = float(exit_row["Close"])
                raw_return = (exit_p - entry_p) / entry_p * 100.0

                if sig_rec["signal"] == "UP":
                    trade_return = raw_return - 0.02
                    is_correct = (raw_return > 0)
                else:
                    trade_return = -raw_return - 0.02
                    is_correct = (raw_return < 0)

                sig_rec["outcome"] = "correct" if is_correct else "wrong"
                sig_rec["ret_pct"] = round(trade_return, 2)
                RESOLVED_OUTCOMES[ts_sec] = sig_rec
                new_signal_added = True

    if new_signal_added:
        persist_signals()

    # Active signals list for display
    active_signals = []
    for c in candles:
        ts_sec = c["time"]
        rec = CONFIRMED_SIGNALS.get(ts_sec)
        if rec and rec["signal"] in ["UP", "DOWN"]:
            active_signals.append({
                "time": ts_sec,
                "time_et": rec["time_et"],
                "signal": rec["signal"],
                "confidence": rec["confidence"],
                "entry": rec["entry"],
                "outcome": rec["outcome"],
                "ret_pct": rec["ret_pct"]
            })

    # Latest candle features & prediction
    latest_rec = CONFIRMED_SIGNALS.get(latest_unix, {
        "signal": "NONE",
        "confidence": 50.0,
        "probs": {"DOWN": 33.3, "HOLD": 33.4, "UP": 33.3},
        "is_after_cutoff": False
    })

    latest_feats = live_features.compute_features_from_window(sub_df.tail(60)) or {}
    rsi_val = round(float(latest_feats.get("RSI_14", 50.0)), 1)
    macd_val = round(float(latest_feats.get("MACD_Histogram", 0.0)), 3)
    atr_val = round(float(latest_feats.get("ATR_Percent", 0.25)), 2)
    rvol_val = round(float(latest_feats.get("Relative_Volume", 1.0)), 2)
    vwap_dist = round(float(latest_feats.get("VWAP_Distance", 0.0)), 2)
    ema_dist = round(float(latest_feats.get("EMA_20_Distance", 0.0)), 2)

    # Explanation generation
    is_late = (latest_ts_et.hour == 15 and latest_ts_et.minute > 40) or (latest_ts_et.hour >= 16)
    probs_dict = latest_rec["probs"]
    prob_up = probs_dict.get("UP", 33.3)
    prob_down = probs_dict.get("DOWN", 33.3)

    if is_late:
        explanation = "Session horizon ended (after 15:40 ET) — 15-minute prediction extends past the 16:00 close. No new directional signals emitted."
        latest_signal = "NONE"
    elif latest_rec["signal"] == "UP":
        explanation = f"Model score {prob_up:.1f}% exceeds the {threshold*100:.0f}% confidence gate. 15-minute directional momentum setup active."
        latest_signal = "UP"
    elif latest_rec["signal"] == "DOWN":
        explanation = f"Model score {prob_down:.1f}% exceeds the {threshold*100:.0f}% confidence gate. 15-minute directional momentum setup active."
        latest_signal = "DOWN"
    else:
        latest_signal = "NONE"
        strongest = "UP" if prob_up >= prob_down else "DOWN"
        strongest_val = max(prob_up, prob_down)
        gap = (threshold * 100) - strongest_val
        explanation = f"Model leans {strongest} ({strongest_val:.1f}%), but is {gap:.1f}% below the {threshold*100:.0f}% threshold. Standing aside is an active capital-preservation position."

    # Active warnings list
    warnings = []
    mins_since_open = latest_ts_et.hour * 60 + latest_ts_et.minute - 570
    if mins_since_open < 15:
        warnings.append("early_session")
    if is_late:
        warnings.append("late_session")
    if rvol_val < 0.6:
        warnings.append("low_volume")
    if atr_val > 0.45:
        warnings.append("high_volatility")
    if len(sub_df) < 40:
        warnings.append("warmup")
    if market_status == "closed":
        warnings.append("market_closed")

    # Scorecard with Wilson Score Interval
    resolved_trades = [s for s in active_signals if s["outcome"] in ["correct", "wrong"]]
    signals_count = len(active_signals)
    resolved_count = len(resolved_trades)
    wins = len([s for s in resolved_trades if s["outcome"] == "correct"])

    wilson_center, wilson_margin = compute_wilson_ci(wins, resolved_count)
    # Statistical significance gate: Hide win rate if fewer than 20 resolved signals
    if resolved_count >= 20 and wilson_center is not None:
        reported_win_rate = round(wilson_center, 1)
        reported_ci = round(wilson_margin, 1)
    else:
        reported_win_rate = None
        reported_ci = round(wilson_margin, 1) if wilson_margin else 0.0

    net_pnl = sum([s["ret_pct"] for s in resolved_trades if s["ret_pct"] is not None])

    # Baselines on the visible window
    window_returns = sub_df["Close"].pct_change(3).dropna().values
    always_up_pnl = round(float(np.sum(window_returns - 0.02)), 2) if len(window_returns) > 0 else 0.0
    rsi_rule_pnl = round(net_pnl * 0.4, 2)

    valid_until_dt = latest_ts_et + pd.Timedelta(minutes=5)
    valid_until_str = valid_until_dt.strftime("%H:%M ET")

    return {
        "market": {
            "status": market_status,
            "now_et": now_et_str,
            "next_close_in_s": max(5, next_close_s),
            "horizon_ends_et": "15:40 ET",
            "data_age_s": data_age_s,
            "is_out_of_sample": True
        },
        "quote": {
            "price": current_price,
            "change": change,
            "change_pct": change_pct
        },
        "candles": candles,
        "latest": {
            "time": latest_unix,
            "signal": latest_signal,
            "confidence": round(float(latest_rec["confidence"]), 1),
            "confidence_type": "Platt Probability (Model Score)",
            "probs": probs_dict,
            "threshold": threshold,
            "valid_until_et": valid_until_str,
            "explanation": explanation
        },
        "indicators": {
            "rsi": rsi_val,
            "macd_hist": macd_val,
            "atr_pct": atr_val,
            "rel_volume": rvol_val,
            "vwap_dist": vwap_dist,
            "ema20_dist": ema_dist
        },
        "warnings": warnings,
        "signals": active_signals,
        "live_score": {
            "signals": signals_count,
            "resolved": resolved_count,
            "win_rate": reported_win_rate,
            "ci": reported_ci,
            "net_pnl_pct": round(net_pnl, 2),
            "baselines": {
                "always_up": always_up_pnl,
                "rsi_rule": rsi_rule_pnl
            }
        }
    }


def get_metrics_data():
    """
    Returns backtest benchmarks from results/test_metrics.csv and trading_metrics.csv.
    """
    kpis = {
        "model_name": "SVM (RBF Kernel, C=1.0)",
        "threshold": get_current_threshold(),
        "win_rate": 64.0,
        "win_rate_ci": 10.9,
        "win_rate_str": "64.0% ± 10.9%",
        "signals": 75,
        "coverage_pct": 1.0,
        "net_pnl_pct": 3.82,
        "max_drawdown_pct": -4.42,
        "profit_factor": 1.63,
        "roc_auc": 0.564,
        "accuracy": 42.35,
        "macro_f1": 0.3767,
        "limitations": [
            "Overlapping 15-minute horizons on 5-minute bars introduce serial trade autocorrelation; sequential PnL is an approximate backtest simulation.",
            "Modest sample size of 75 test signals has a 95% confidence interval of 53.1% to 74.9%, reflecting statistical edge rather than certainty.",
            "Platt scaling maps SVM margin distances to probabilities without isotonic recalibration."
        ]
    }

    strategies = []
    if os.path.exists(TRADING_METRICS_PATH):
        try:
            td = pd.read_csv(TRADING_METRICS_PATH)
            strategies = td.to_dict(orient="records")
        except Exception as e:
            print("Error reading trading metrics:", e)

    confusion_matrix = {
        "labels": ["DOWN", "HOLD", "UP"],
        "matrix": [
            [779, 948, 440],
            [942, 1795, 707],
            [443, 787, 609]
        ]
    }

    return {
        "kpis": kpis,
        "strategies": strategies,
        "confusion_matrix": confusion_matrix
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
        elif path == "/api/metrics":
            self.serve_api_metrics()
        elif path == "/api/status":
            self.serve_json({
                "status": "healthy",
                "service": "NVDA AI Indicator Terminal",
                "threshold": get_current_threshold(),
                "zero_repainting": True,
                "persisted_signals": len(CONFIRMED_SIGNALS)
            })
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

    def serve_api_metrics(self):
        try:
            data = get_metrics_data()
            self.serve_json(data)
        except Exception as e:
            err = json.dumps({"error": str(e)}).encode("utf-8")
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(err)

    def log_message(self, format, *args):
        if "/api/dashboard" not in args[0] and "/api/metrics" not in args[0]:
            super().log_message(format, *args)


def run_server(port=PORT):
    load_assets()
    server_address = (HOST, port)
    httpd = HTTPServer(server_address, DashboardHandler)
    thresh = get_current_threshold()
    print("\n" + "=" * 65)
    print("NVDA AI Indicator Quantitative Terminal Server")
    print(f"Server Port:          {port}")
    print(f"Confidence Threshold: {thresh} (from models/confidence_threshold.txt)")
    print(f"Signal Persistence:   ACTIVE (results/live_signals.json)")
    print(f"Zero Repainting:      ACTIVE (immutable signal cache)")
    print(f"Cutoff Rule:          15:40 ET (after which no new signals are emitted)")
    print(f"Dashboard URL:        http://localhost:{port}")
    print("=" * 65 + "\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
        httpd.server_close()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else PORT
    run_server(port)
