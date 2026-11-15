
import os
import numpy as np
import pandas as pd

# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = "data/processed/NVDA_5min_cleaned.csv"
OUTPUT_DIR = "data/processed"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "NVDA_features.csv")

# Prediction horizon: 15 minutes = 3 five-minute candles
PREDICTION_CANDLES = 3

# Target thresholds
UP_THRESHOLD = 0.15
DOWN_THRESHOLD = -0.15

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 1. LOAD AND VALIDATE DATA
# ============================================================

print("\nLoading cleaned NVDA data...")

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(
        f"Input file not found: {INPUT_FILE}"
    )

df = pd.read_csv(INPUT_FILE)

print("Original dataset shape:", df.shape)

required_columns = [
    "Timestamp",
    "Open",
    "High",
    "Low",
    "Close",
    "Volume"
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

# Parse timestamps as UTC
df["Timestamp"] = pd.to_datetime(
    df["Timestamp"],
    utc=True,
    errors="coerce"
)

# Always derive New York time from UTC.
# This also handles daylight-saving-time changes.
df["Timestamp_ET"] = (
    df["Timestamp"].dt.tz_convert("America/New_York")
)

df = df.dropna(
    subset=["Timestamp", "Timestamp_ET"]
).copy()

# Convert numerical columns
numeric_columns = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume"
]

for col in ["VWAP", "Transactions"]:
    if col in df.columns:
        numeric_columns.append(col)

for col in numeric_columns:
    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )

# Replace infinite values
df.replace(
    [np.inf, -np.inf],
    np.nan,
    inplace=True
)

# Remove rows with missing essential OHLCV values
df.dropna(
    subset=[
        "Open",
        "High",
        "Low",
        "Close",
        "Volume"
    ],
    inplace=True
)

# Remove invalid prices and negative volume
df = df[
    (df["Open"] > 0) &
    (df["High"] > 0) &
    (df["Low"] > 0) &
    (df["Close"] > 0) &
    (df["Volume"] >= 0)
].copy()

# Validate OHLC relationships
valid_ohlc = (
    (df["High"] >= df["Low"]) &
    (df["High"] >= df["Open"]) &
    (df["High"] >= df["Close"]) &
    (df["Low"] <= df["Open"]) &
    (df["Low"] <= df["Close"])
)

df = df[valid_ohlc].copy()

# Remove duplicate timestamps
df.drop_duplicates(
    subset=["Timestamp"],
    keep="first",
    inplace=True
)

# Sort chronologically
df = df.sort_values(
    "Timestamp"
).reset_index(drop=True)

# Keep only regular US market session
# Monday-Friday, 9:30 AM to before 4:00 PM ET
local_time = df["Timestamp_ET"]

weekday_mask = local_time.dt.dayofweek < 5

session_mask = (
    (local_time.dt.hour * 60 + local_time.dt.minute >= 570) &
    (local_time.dt.hour * 60 + local_time.dt.minute < 960)
)

df = df[
    weekday_mask & session_mask
].copy()

df = df.sort_values(
    "Timestamp"
).reset_index(drop=True)

# Create trading date for session-wise calculations
df["Trading_Date"] = df[
    "Timestamp_ET"
].dt.date

# Check for duplicate timestamps again
df.drop_duplicates(
    subset=["Timestamp"],
    keep="first",
    inplace=True
)

df = df.sort_values(
    "Timestamp"
).reset_index(drop=True)

print("Cleaned dataset shape:", df.shape)
print(
    "Date range:",
    df["Timestamp_ET"].min(),
    "to",
    df["Timestamp_ET"].max()
)


# ============================================================
# 2. PRICE-BASED FEATURES
# ============================================================

print("\nCreating price-based features...")

grouped_close = df.groupby(
    "Trading_Date"
)["Close"]

# Percentage returns over different lookback periods
for period in [1, 3, 6, 12, 24]:

    df[f"Return_{period}"] = (
        grouped_close.pct_change(period) * 100
    )

# High-low price range
df["High_Low_Range"] = (
    (df["High"] - df["Low"]) /
    df["Close"] * 100
)

# Intracandle close-open return
df["Close_Open_Return"] = (
    (df["Close"] - df["Open"]) /
    df["Open"] * 100
)

# Distance from VWAP
if "VWAP" in df.columns:

    df["VWAP_Distance"] = (
        (df["Close"] - df["VWAP"]) /
        df["VWAP"].replace(0, np.nan) * 100
    )


# ============================================================
# 3. EXPONENTIAL MOVING AVERAGES
# ============================================================

print("Creating EMA features...")

for period in [9, 20, 50]:

    ema = df.groupby(
        "Trading_Date"
    )["Close"].transform(
        lambda x: x.ewm(
            span=period,
            adjust=False,
            min_periods=period
        ).mean()
    )

    df[f"EMA_{period}_Distance"] = (
        (df["Close"] - ema) /
        ema.replace(0, np.nan) * 100
    )


# ============================================================
# 4. RSI - RELATIVE STRENGTH INDEX
# ============================================================

print("Calculating RSI...")

def calculate_rsi(series, period=14):

    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    average_gain = gain.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period
    ).mean()

    average_loss = loss.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period
    ).mean()

    rs = (
        average_gain /
        average_loss.replace(0, np.nan)
    )

    rsi = 100 - (100 / (1 + rs))

    # No losses: RSI = 100
    rsi = rsi.mask(
        (average_loss == 0) &
        (average_gain > 0),
        100
    )

    # No gains and no losses: RSI = 50
    rsi = rsi.mask(
        (average_loss == 0) &
        (average_gain == 0),
        50
    )

    return rsi


df["RSI_14"] = df.groupby(
    "Trading_Date"
)["Close"].transform(
    calculate_rsi
)


# ============================================================
# 5. MACD
# ============================================================

print("Calculating MACD...")

grouped_close = df.groupby(
    "Trading_Date"
)["Close"]

ema12 = grouped_close.transform(
    lambda x: x.ewm(
        span=12,
        adjust=False,
        min_periods=26
    ).mean()
)

ema26 = grouped_close.transform(
    lambda x: x.ewm(
        span=26,
        adjust=False,
        min_periods=26
    ).mean()
)

macd = ema12 - ema26

signal = macd.groupby(
    df["Trading_Date"]
).transform(
    lambda x: x.ewm(
        span=9,
        adjust=False,
        min_periods=9
    ).mean()
)

# Normalize MACD by closing price
df["MACD"] = (
    macd / df["Close"] * 100
)

df["MACD_Signal"] = (
    signal / df["Close"] * 100
)

df["MACD_Histogram"] = (
    (macd - signal) / df["Close"] * 100
)


# ============================================================
# 6. BOLLINGER BANDS
# ============================================================

print("Calculating Bollinger Bands...")

rolling_mean = df.groupby(
    "Trading_Date"
)["Close"].transform(
    lambda x: x.rolling(
        window=20,
        min_periods=20
    ).mean()
)

rolling_std = df.groupby(
    "Trading_Date"
)["Close"].transform(
    lambda x: x.rolling(
        window=20,
        min_periods=20
    ).std()
)

upper_band = rolling_mean + 2 * rolling_std
lower_band = rolling_mean - 2 * rolling_std

df["BB_Width"] = (
    (upper_band - lower_band) /
    rolling_mean.replace(0, np.nan) * 100
)

df["BB_Percent_B"] = (
    (df["Close"] - lower_band) /
    (upper_band - lower_band).replace(0, np.nan)
)


# ============================================================
# 7. ATR - AVERAGE TRUE RANGE
# ============================================================

print("Calculating ATR...")

previous_close = df.groupby(
    "Trading_Date"
)["Close"].shift(1)

tr1 = df["High"] - df["Low"]

tr2 = (
    df["High"] - previous_close
).abs()

tr3 = (
    df["Low"] - previous_close
).abs()

df["True_Range"] = pd.concat(
    [tr1, tr2, tr3],
    axis=1
).max(axis=1)

df["ATR_14"] = df.groupby(
    "Trading_Date"
)["True_Range"].transform(
    lambda x: x.ewm(
        alpha=1 / 14,
        adjust=False,
        min_periods=14
    ).mean()
)

df["ATR_Percent"] = (
    df["ATR_14"] /
    df["Close"] * 100
)


# ============================================================
# 8. VOLUME FEATURES
# ============================================================

print("Creating volume features...")

# Log-transformed volume
df["Log_Volume"] = np.log1p(
    df["Volume"]
)

# Session-wise rolling average volume
rolling_volume_mean = df.groupby(
    "Trading_Date"
)["Volume"].transform(
    lambda x: x.rolling(
        window=20,
        min_periods=5
    ).mean()
)

# Relative volume
df["Relative_Volume"] = (
    df["Volume"] /
    rolling_volume_mean.replace(0, np.nan)
)

# Percentage change in volume
df["Volume_Change"] = df.groupby(
    "Trading_Date"
)["Volume"].transform(
    lambda x: x.pct_change() * 100
)

# Log-transformed transaction count
if "Transactions" in df.columns:

    df["Transactions"] = pd.to_numeric(
        df["Transactions"],
        errors="coerce"
    )

    df["Log_Transactions"] = np.log1p(
        df["Transactions"].clip(lower=0)
    )

# On Balance Volume (OBV)
price_direction = df.groupby(
    "Trading_Date"
)["Close"].diff()

signed_volume = (
    np.sign(price_direction) *
    df["Volume"]
)

df["OBV"] = signed_volume.groupby(
    df["Trading_Date"]
).cumsum()

# Normalize OBV using session-wise rolling averages
obv_mean = df.groupby(
    "Trading_Date"
)["OBV"].transform(
    lambda x: x.rolling(
        window=20,
        min_periods=5
    ).mean()
)

rolling_volume_mean = df.groupby(
    "Trading_Date"
)["Volume"].transform(
    lambda x: x.rolling(
        window=20,
        min_periods=5
    ).mean()
)

df["OBV_Normalized"] = (
    (df["OBV"] - obv_mean) /
    rolling_volume_mean.replace(0, np.nan)
)


# ============================================================
# 9. TIME-BASED FEATURES
# ============================================================

print("Creating time features...")

local_time = df["Timestamp_ET"]

minutes_since_open = (
    local_time.dt.hour * 60 +
    local_time.dt.minute - 570
)

df["Minutes_Since_Open"] = minutes_since_open

df["Hour"] = local_time.dt.hour

df["Day_Of_Week"] = (
    local_time.dt.dayofweek
)

# Cyclical encoding of time within the trading session
df["Time_Sin"] = np.sin(
    2 * np.pi * minutes_since_open / 390
)

df["Time_Cos"] = np.cos(
    2 * np.pi * minutes_since_open / 390
)

# Cyclical encoding of weekday
df["Weekday_Sin"] = np.sin(
    2 * np.pi * df["Day_Of_Week"] / 5
)

df["Weekday_Cos"] = np.cos(
    2 * np.pi * df["Day_Of_Week"] / 5
)


# ============================================================
# 10. TARGET VARIABLE CREATION
# ============================================================

print("Creating target variable...")

# Future closing price after exactly 3 candles
df["Future_Close"] = df.groupby(
    "Trading_Date"
)["Close"].shift(
    -PREDICTION_CANDLES
)

# Timestamp of the future candle
df["Future_Timestamp"] = df.groupby(
    "Trading_Date"
)["Timestamp_ET"].shift(
    -PREDICTION_CANDLES
)

# Preserve target end timestamp for leakage-free splitting
df["Target_End_Timestamp"] = (
    df["Future_Timestamp"]
)

# Verify that the future candle is exactly 15 minutes later
valid_target = (
    df["Future_Timestamp"] -
    df["Timestamp_ET"]
) == pd.Timedelta(minutes=15)

# Calculate future percentage return
df["Future_Return"] = (
    (df["Future_Close"] - df["Close"]) /
    df["Close"] * 100
)

# Assign target classes:
# DOWN = 0
# HOLD = 1
# UP = 2

df["Target"] = np.select(
    [
        df["Future_Return"] < DOWN_THRESHOLD,
        df["Future_Return"] > UP_THRESHOLD
    ],
    [0, 2],
    default=1
)

# Invalidate targets that do not have a valid future candle
df.loc[
    ~valid_target,
    "Target"
] = np.nan


# ============================================================
# 11. FINAL CLEANING
# ============================================================

print("Removing invalid target rows...")

# Replace infinite values
df.replace(
    [np.inf, -np.inf],
    np.nan,
    inplace=True
)

# Keep only rows with a valid target
df.dropna(
    subset=[
        "Target",
        "Target_End_Timestamp",
        "Future_Return"
    ],
    inplace=True
)

df["Target"] = df["Target"].astype(int)

# Remove helper columns.
# Target_End_Timestamp and Future_Return are deliberately kept.
df.drop(
    columns=[
        "Trading_Date",
        "True_Range",
        "Future_Close",
        "Future_Timestamp"
    ],
    inplace=True,
    errors="ignore"
)

df = df.sort_values(
    "Timestamp"
).reset_index(drop=True)


# ============================================================
# 12. FEATURE SUMMARY AND SAVE
# ============================================================

print("\nPreparing final dataset...")

# Model input features
FEATURES = [
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
    "Weekday_Cos"
]

# Optional features
if "VWAP_Distance" in df.columns:
    FEATURES.append("VWAP_Distance")

if "Log_Transactions" in df.columns:
    FEATURES.append("Log_Transactions")

# Check that all expected features were created
missing_features = [
    feature for feature in FEATURES
    if feature not in df.columns
]

if missing_features:
    raise ValueError(
        f"Feature creation failed: {missing_features}"
    )

print("\nNumber of model features:", len(FEATURES))

# Save complete dataset, including timestamps and evaluation columns.
# Training code should select FEATURES explicitly.
df.to_csv(
    OUTPUT_FILE,
    index=False
)

# ============================================================
# 13. DISPLAY RESULTS
# ============================================================

print("\n==========================================")
print("FEATURE ENGINEERING COMPLETED SUCCESSFULLY")
print("==========================================")

print("\nFinal dataset shape:", df.shape)

print("\nFeatures used by the models:")

for i, feature in enumerate(FEATURES, start=1):
    print(f"{i}. {feature}")

print("\nTarget distribution:")

target_counts = (
    df["Target"]
    .value_counts()
    .reindex([0, 1, 2], fill_value=0)
)

print(
    target_counts.rename(
        index={
            0: "DOWN",
            1: "HOLD",
            2: "UP"
        }
    )
)

print("\nTarget percentages:")

print(
    (
        target_counts / target_counts.sum() * 100
    ).round(2).rename(
        index={
            0: "DOWN",
            1: "HOLD",
            2: "UP"
        }
    )
)

print("\nMissing values in model features:")

print(
    df[FEATURES].isnull().sum()
)

print("\nSaved feature dataset at:")
print(OUTPUT_FILE)

print("\nRemember:")
print("- Target_End_Timestamp is for chronological splitting.")
print("- Future_Return is for evaluation, not model input.")
print("- Target is the prediction label.")
print("- Do not include timestamps or future information as model features.")
