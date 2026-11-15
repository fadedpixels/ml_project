"""
live_features.py - Unified Feature Engineering & Inference Module
Ensures exact consistency between historical training and live streaming inference.
Inspired by best practices from quantitative intraday trading (e.g. Pyligent/algotrading-live).
"""

import numpy as np
import pandas as pd

# The 28 stationary features used by the models
FEATURE_NAMES = [
    "Return_1",
    "Return_3",
    "Return_6",
    "Return_12",
    "Return_24",
    "High_Low_Range",
    "Close_Open_Return",
    "EMA_9_Distance",
    "EMA_20_Distance",
    "EMA_50_Distance",
    "RSI_14",
    "MACD",
    "MACD_Signal",
    "MACD_Histogram",
    "BB_Width",
    "BB_Percent_B",
    "ATR_Percent",
    "Log_Volume",
    "Relative_Volume",
    "Volume_Change",
    "OBV_Normalized",
    "Minutes_Since_Open",
    "Time_Sin",
    "Time_Cos",
    "Weekday_Sin",
    "Weekday_Cos",
    "VWAP_Distance",
    "Log_Transactions"
]

CLASS_NAMES = ["DOWN", "HOLD", "UP"]


def compute_rsi(series, period=14):
    """Computes relative strength index (RSI)."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=1).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=1).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)


def compute_features_from_window(df_window):
    """
    Computes all 28 model features for the latest closed candle in a DataFrame window.
    df_window must contain: Open, High, Low, Close, Volume, Timestamp_ET (or Timestamp),
    and optionally VWAP, Transactions.
    """
    if len(df_window) < 5:
        return None

    df = df_window.copy().reset_index(drop=True)
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Time-based features
    if "Timestamp_ET" in df.columns:
        ts = pd.to_datetime(df["Timestamp_ET"].iloc[-1])
    else:
        ts = pd.to_datetime(df["Timestamp"].iloc[-1], utc=True).tz_convert("America/New_York")

    mins = ts.hour * 60 + ts.minute - 570  # Minutes from 9:30 AM
    day_of_week = ts.dayofweek

    features = {}

    # 1. Returns over lookback periods
    close_s = df["Close"]
    for p in [1, 3, 6, 12, 24]:
        if len(close_s) > p and close_s.iloc[-1 - p] > 0:
            features[f"Return_{p}"] = float((close_s.iloc[-1] - close_s.iloc[-1 - p]) / close_s.iloc[-1 - p] * 100)
        else:
            features[f"Return_{p}"] = 0.0

    # 2. Intracandle features
    features["High_Low_Range"] = float((df["High"].iloc[-1] - df["Low"].iloc[-1]) / df["Close"].iloc[-1] * 100)
    features["Close_Open_Return"] = float((df["Close"].iloc[-1] - df["Open"].iloc[-1]) / df["Open"].iloc[-1] * 100)

    # 3. EMA Distances
    for span in [9, 20, 50]:
        ema_val = close_s.ewm(span=span, adjust=False, min_periods=1).mean().iloc[-1]
        features[f"EMA_{span}_Distance"] = float((close_s.iloc[-1] - ema_val) / ema_val * 100) if ema_val > 0 else 0.0

    # 4. RSI
    features["RSI_14"] = float(compute_rsi(close_s, period=14).iloc[-1])

    # 5. MACD
    ema12 = close_s.ewm(span=12, adjust=False, min_periods=1).mean()
    ema26 = close_s.ewm(span=26, adjust=False, min_periods=1).mean()
    macd_raw = ema12 - ema26
    signal_raw = macd_raw.ewm(span=9, adjust=False, min_periods=1).mean()
    c_last = close_s.iloc[-1]

    features["MACD"] = float(macd_raw.iloc[-1] / c_last * 100)
    features["MACD_Signal"] = float(signal_raw.iloc[-1] / c_last * 100)
    features["MACD_Histogram"] = float((macd_raw.iloc[-1] - signal_raw.iloc[-1]) / c_last * 100)

    # 6. Bollinger Bands
    bb_mean = close_s.rolling(20, min_periods=1).mean().iloc[-1]
    bb_std = close_s.rolling(20, min_periods=1).std().iloc[-1]
    bb_std = 0.0 if np.isnan(bb_std) else bb_std
    bb_width = (4 * bb_std) / bb_mean * 100 if bb_mean > 0 else 0.0
    features["BB_Width"] = float(bb_width)
    features["BB_Percent_B"] = float((close_s.iloc[-1] - (bb_mean - 2 * bb_std)) / (4 * bb_std)) if bb_std > 0 else 0.5

    # 7. ATR
    prev_close = close_s.shift(1)
    tr1 = df["High"] - df["Low"]
    tr2 = (df["High"] - prev_close).abs()
    tr3 = (df["Low"] - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / 14, adjust=False, min_periods=1).mean().iloc[-1]
    features["ATR_Percent"] = float(atr / c_last * 100) if c_last > 0 else 0.0

    # 8. Volume Features
    vol_s = df["Volume"]
    features["Log_Volume"] = float(np.log1p(max(0, vol_s.iloc[-1])))
    roll_vol = vol_s.rolling(20, min_periods=1).mean().iloc[-1]
    features["Relative_Volume"] = float(vol_s.iloc[-1] / roll_vol) if roll_vol > 0 else 1.0
    features["Volume_Change"] = float(vol_s.pct_change().iloc[-1] * 100) if len(vol_s) > 1 and not np.isnan(vol_s.pct_change().iloc[-1]) else 0.0

    # OBV Normalized
    pdir = np.sign(close_s.diff().fillna(0))
    obv = (pdir * vol_s).cumsum()
    obv_mean = obv.rolling(20, min_periods=1).mean().iloc[-1]
    features["OBV_Normalized"] = float((obv.iloc[-1] - obv_mean) / roll_vol) if roll_vol > 0 else 0.0

    # 9. Time features
    features["Minutes_Since_Open"] = int(mins)
    features["Time_Sin"] = float(np.sin(2 * np.pi * mins / 390))
    features["Time_Cos"] = float(np.cos(2 * np.pi * mins / 390))
    features["Weekday_Sin"] = float(np.sin(2 * np.pi * day_of_week / 5))
    features["Weekday_Cos"] = float(np.cos(2 * np.pi * day_of_week / 5))

    # Optional VWAP & Transactions
    if "VWAP" in df.columns and pd.notna(df["VWAP"].iloc[-1]) and float(df["VWAP"].iloc[-1]) > 0:
        vwap = float(df["VWAP"].iloc[-1])
        features["VWAP_Distance"] = float((c_last - vwap) / vwap * 100)
    else:
        features["VWAP_Distance"] = 0.0

    if "Transactions" in df.columns and pd.notna(df["Transactions"].iloc[-1]):
        features["Log_Transactions"] = float(np.log1p(max(0, float(df["Transactions"].iloc[-1]))))
    else:
        features["Log_Transactions"] = 0.0

    return features


def predict_candle(model, feature_names, features_dict, confidence_threshold=0.40):
    """
    Performs calibrated inference with a confidence threshold.
    Returns:
        signal (str): 'UP', 'DOWN', or 'HOLD' (No Clear Signal)
        confidence (float): probability percentage
        probabilities (dict): probabilities for DOWN, HOLD, UP
        is_confident (bool): whether the signal met the confidence gate
    """
    if model is None or feature_names is None or features_dict is None:
        return "HOLD", 50.0, {"DOWN": 0.33, "HOLD": 0.34, "UP": 0.33}, False

    try:
        # Create single-row DataFrame matching the exact training columns
        row_dict = {f: [features_dict.get(f, 0.0)] for f in feature_names}
        input_df = pd.DataFrame(row_dict)

        if isinstance(model, dict):
            # Soft voting ensemble
            probs_list = [m.predict_proba(input_df)[0] for m in model.values()]
            probs = np.mean(probs_list, axis=0)
        else:
            probs = model.predict_proba(input_df)[0]

        prob_down, prob_hold, prob_up = float(probs[0]), float(probs[1]), float(probs[2])
        prob_dict = {
            "DOWN": round(prob_down * 100, 1),
            "HOLD": round(prob_hold * 100, 1),
            "UP": round(prob_up * 100, 1)
        }

        # Confidence Gate (Meta-Filter):
        # A directional signal is ONLY fired when model conviction exceeds confidence_threshold
        if prob_up >= confidence_threshold and prob_up > prob_down:
            return "UP", round(prob_up * 100, 1), prob_dict, True
        elif prob_down >= confidence_threshold and prob_down > prob_up:
            return "DOWN", round(prob_down * 100, 1), prob_dict, True
        else:
            # Below confidence gate -> Capital preservation / No Clear Signal
            max_conf = max(prob_hold, prob_up, prob_down)
            return "HOLD", round(max_conf * 100, 1), prob_dict, False

    except Exception as e:
        print("Inference error in live_features:", e)
        return "HOLD", 50.0, {"DOWN": 0.33, "HOLD": 0.34, "UP": 0.33}, False
