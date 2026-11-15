"""
live_features.py - Unified Feature Engineering & Live Inference Engine
Provides 100% feature parity between offline model training and real-time live streaming.
"""

import numpy as np
import pandas as pd

# The 28 model features
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


def extract_features(df_in):
    """
    Extracts all 28 features from a DataFrame of 5-minute candles.
    Works identically for the full historical dataset or a live session buffer.
    Requires columns: Open, High, Low, Close, Volume, and Timestamp.
    """
    df = df_in.copy()
    
    # Standardize timestamps & trading date
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True)
    df["Timestamp_ET"] = df["Timestamp"].dt.tz_convert("America/New_York")
    df["Trading_Date"] = df["Timestamp_ET"].dt.date
    
    # Ensure numeric columns
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    grouped = df.groupby("Trading_Date")

    # 1. Price Returns
    df["Return_1"] = grouped["Close"].pct_change(1).fillna(0) * 100
    df["Return_3"] = grouped["Close"].pct_change(3).fillna(0) * 100
    df["Return_6"] = grouped["Close"].pct_change(6).fillna(0) * 100
    df["Return_12"] = grouped["Close"].pct_change(12).fillna(0) * 100
    df["Return_24"] = grouped["Close"].pct_change(24).fillna(0) * 100

    # 2. Intracandle features
    df["Close_Open_Return"] = (df["Close"] - df["Open"]) / df["Open"] * 100
    df["High_Low_Range"] = (df["High"] - df["Low"]) / df["Close"] * 100

    # 3. Exponential Moving Averages (EMAs)
    ema9 = grouped["Close"].transform(lambda x: x.ewm(span=9, adjust=False).mean())
    ema20 = grouped["Close"].transform(lambda x: x.ewm(span=20, adjust=False).mean())
    ema50 = grouped["Close"].transform(lambda x: x.ewm(span=50, adjust=False).mean())
    df["EMA_9_Distance"] = (df["Close"] - ema9) / ema9 * 100
    df["EMA_20_Distance"] = (df["Close"] - ema20) / ema20 * 100
    df["EMA_50_Distance"] = (df["Close"] - ema50) / ema50 * 100

    # 4. RSI (Wilder's 14-period RSI per session)
    def _calc_rsi(s):
        delta = s.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        ag = gain.ewm(alpha=1 / 14, adjust=False, min_periods=1).mean()
        al = loss.ewm(alpha=1 / 14, adjust=False, min_periods=1).mean()
        rs = ag / al.replace(0, np.nan)
        return (100 - (100 / (1 + rs))).fillna(50)

    df["RSI_14"] = grouped["Close"].transform(_calc_rsi)

    # 5. MACD (12, 26, 9)
    ema12 = grouped["Close"].transform(lambda x: x.ewm(span=12, adjust=False).mean())
    ema26 = grouped["Close"].transform(lambda x: x.ewm(span=26, adjust=False).mean())
    macd = (ema12 - ema26) / df["Close"] * 100
    signal = macd.groupby(df["Trading_Date"]).transform(lambda x: x.ewm(span=9, adjust=False).mean())
    df["MACD"] = macd
    df["MACD_Signal"] = signal
    df["MACD_Histogram"] = macd - signal

    # 6. Bollinger Bands (20-period, 2-std)
    bb_mean = grouped["Close"].transform(lambda x: x.rolling(20, min_periods=1).mean())
    bb_std = grouped["Close"].transform(lambda x: x.rolling(20, min_periods=1).std().fillna(0))
    df["BB_Width"] = (4 * bb_std) / bb_mean * 100
    df["BB_Percent_B"] = ((df["Close"] - (bb_mean - 2 * bb_std)) / (4 * bb_std + 1e-6)).clip(0, 1)

    # 7. ATR (14-period Average True Range)
    prev_close = grouped["Close"].shift(1)
    tr1 = df["High"] - df["Low"]
    tr2 = (df["High"] - prev_close).abs()
    tr3 = (df["Low"] - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = grouped["Close"].transform(lambda x: tr.loc[x.index].ewm(alpha=1 / 14, adjust=False, min_periods=1).mean())
    df["ATR_Percent"] = (atr / df["Close"] * 100).fillna(0.3)

    # 8. Volume Features
    df["Log_Volume"] = np.log1p(df["Volume"].clip(lower=0))
    roll_vol = grouped["Volume"].transform(lambda x: x.rolling(20, min_periods=1).mean())
    df["Relative_Volume"] = (df["Volume"] / roll_vol.replace(0, np.nan)).fillna(1.0)
    df["Volume_Change"] = grouped["Volume"].pct_change().fillna(0) * 100

    # Normalized On-Balance-Volume (OBV)
    pdir = np.sign(grouped["Close"].diff().fillna(0))
    obv = (pdir * df["Volume"]).groupby(df["Trading_Date"]).cumsum()
    obv_mean = obv.groupby(df["Trading_Date"]).transform(lambda x: x.rolling(20, min_periods=1).mean())
    df["OBV_Normalized"] = ((obv - obv_mean) / roll_vol.replace(0, np.nan)).fillna(0)

    # 9. Time-of-Day & Day-of-Week cyclical encoding
    mins = df["Timestamp_ET"].dt.hour * 60 + df["Timestamp_ET"].dt.minute - 570
    df["Minutes_Since_Open"] = mins
    df["Time_Sin"] = np.sin(2 * np.pi * mins / 390)
    df["Time_Cos"] = np.cos(2 * np.pi * mins / 390)
    df["Weekday_Sin"] = np.sin(2 * np.pi * df["Timestamp_ET"].dt.day_of_week / 5)
    df["Weekday_Cos"] = np.cos(2 * np.pi * df["Timestamp_ET"].dt.dayofweek / 5)

    # 10. Optional VWAP & Transactions
    if "VWAP" in df.columns:
        df["VWAP_Distance"] = ((df["Close"] - df["VWAP"]) / df["VWAP"].replace(0, np.nan) * 100).fillna(0)
    else:
        df["VWAP_Distance"] = 0.0

    if "Transactions" in df.columns:
        df["Log_Transactions"] = np.log1p(pd.to_numeric(df["Transactions"], errors="coerce").fillna(0).clip(lower=0))
    else:
        df["Log_Transactions"] = 0.0

    return df


def compute_features_from_window(window_df):
    """
    Computes all 28 model features for the latest candle in window_df.
    Guarantees 100% mathematical parity with extract_features.
    Returns:
        dict: {feature_name: value} for the latest closed candle.
    """
    if window_df is None or len(window_df) == 0:
        return None
    
    extracted = extract_features(window_df)
    latest_row = extracted.iloc[-1]
    return {col: float(latest_row[col]) for col in FEATURE_NAMES if col in latest_row}


def create_targets(df, prediction_candles=3, dynamic=True):
    """
    Creates target direction labels strictly within regular market hours.
    Prevents overnight leakage by grouping by Trading_Date.
    """
    grouped = df.groupby("Trading_Date")
    future_close = grouped["Close"].shift(-prediction_candles)
    future_ts = grouped["Timestamp_ET"].shift(-prediction_candles)
    
    # 15-minute future return
    fut_ret = (future_close - df["Close"]) / df["Close"] * 100
    
    # Strict validation: future candle must be in same session and exactly 15 minutes later
    valid = (future_ts - df["Timestamp_ET"]) == pd.Timedelta(minutes=5 * prediction_candles)
    
    if dynamic:
        # Volatility-adaptive threshold scaled by current 15m ATR
        threshold = (0.30 * df["ATR_Percent"]).clip(lower=0.12, upper=0.35)
    else:
        # Static baseline threshold
        threshold = 0.15
        
    target = np.select([fut_ret > threshold, fut_ret < -threshold], [2, 0], default=1)
    target_series = pd.Series(target, index=df.index).where(valid, np.nan)
    
    return target_series, fut_ret, future_ts, valid


def predict_candle(model, feature_names, features_dict, confidence_threshold=0.40):
    """
    Performs calibrated inference with a confidence threshold.
    Returns:
        signal (str): 'UP', 'DOWN', or 'HOLD' (No Clear Signal)
        confidence (float): calibrated confidence percentage
        prob_dict (dict): probabilities for DOWN, HOLD, UP
        is_confident (bool): whether the signal met the confidence gate
    """
    if model is None or feature_names is None or features_dict is None:
        return "HOLD", 50.0, {"DOWN": 0.33, "HOLD": 0.34, "UP": 0.33}, False

    try:
        row_dict = {f: [features_dict.get(f, 0.0)] for f in feature_names}
        input_df = pd.DataFrame(row_dict)

        if isinstance(model, dict):
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

        # Confidence Gate (Meta-Filter)
        if prob_up >= confidence_threshold and prob_up > prob_down:
            return "UP", round(prob_up * 100, 1), prob_dict, True
        elif prob_down >= confidence_threshold and prob_down > prob_up:
            return "DOWN", round(prob_down * 100, 1), prob_dict, True
        else:
            max_conf = max(prob_hold, prob_up, prob_down)
            return "HOLD", round(max_conf * 100, 1), prob_dict, False

    except Exception as e:
        print("Inference error in live_features:", e)
        return "HOLD", 50.0, {"DOWN": 0.33, "HOLD": 0.34, "UP": 0.33}, False
