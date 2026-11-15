
import os
import numpy as np
import pandas as pd

# ==========================================
# 1. CONFIGURATION
# ==========================================

INPUT_FILE = "data/raw/NVDA_5min_regular_2y.csv"
OUTPUT_DIR = "data/processed"

CLEANED_FILE = os.path.join(
    OUTPUT_DIR, "NVDA_5min_cleaned.csv"
)

REPORT_FILE = os.path.join(
    OUTPUT_DIR, "preprocessing_report.txt"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==========================================
# 2. LOAD DATASET
# ==========================================

print("\n========== DATA LOADING ==========")

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(
        f"Dataset not found: {INPUT_FILE}"
    )

df = pd.read_csv(INPUT_FILE)

initial_rows = len(df)

print("Dataset loaded successfully!")
print("Initial shape:", df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nFirst five records:")
print(df.head())


# ==========================================
# 3. STANDARDIZE COLUMN NAMES
# ==========================================

df.columns = df.columns.str.strip()

# Handle common alternative column names
rename_map = {
    "t": "Timestamp",
    "o": "Open",
    "h": "High",
    "l": "Low",
    "c": "Close",
    "v": "Volume",
    "vw": "VWAP",
    "n": "Transactions"
}

df.rename(columns=rename_map, inplace=True)

required_columns = [
    "Timestamp", "Open", "High",
    "Low", "Close", "Volume"
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Required columns missing: {missing_columns}"
    )


# ==========================================
# 4. TIMESTAMP PROCESSING
# ==========================================

print("\n========== TIMESTAMP PROCESSING ==========")

# Parse UTC timestamps
df["Timestamp"] = pd.to_datetime(
    df["Timestamp"],
    utc=True,
    errors="coerce"
)

# Remove rows with invalid timestamps
invalid_timestamps = df["Timestamp"].isna().sum()

df.dropna(subset=["Timestamp"], inplace=True)

# Convert to New York time
df["Timestamp_ET"] = (
    df["Timestamp"].dt.tz_convert("America/New_York")
)

# Remove duplicate timestamps
duplicates = df.duplicated(
    subset=["Timestamp"],
    keep="first"
).sum()

df.drop_duplicates(
    subset=["Timestamp"],
    keep="first",
    inplace=True
)

# Sort chronologically
df.sort_values("Timestamp", inplace=True)

df.reset_index(drop=True, inplace=True)


# ==========================================
# 5. NUMERIC DATA CLEANING
# ==========================================

print("\n========== NUMERIC DATA CLEANING ==========")

numeric_columns = [
    "Open", "High", "Low",
    "Close", "Volume"
]

optional_numeric = [
    "VWAP", "Transactions"
]

for col in numeric_columns + optional_numeric:
    if col in df.columns:
        df[col] = pd.to_numeric(
            df[col], errors="coerce"
        )

# Replace infinite values with NaN
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Remove rows missing essential market information
missing_essential = df[numeric_columns].isna().any(axis=1).sum()

df.dropna(
    subset=numeric_columns,
    inplace=True
)

# Remove negative volume
negative_volume = (df["Volume"] < 0).sum()

df = df[df["Volume"] >= 0].copy()

# Remove negative transaction counts, if available
if "Transactions" in df.columns:
    df = df[
        df["Transactions"].isna()
        | (df["Transactions"] >= 0)
    ].copy()


# ==========================================
# 6. OHLC VALIDATION
# ==========================================

print("\n========== OHLC VALIDATION ==========")

invalid_ohlc = (
    (df["High"] < df["Low"]) |
    (df["High"] < df["Open"]) |
    (df["High"] < df["Close"]) |
    (df["Low"] > df["Open"]) |
    (df["Low"] > df["Close"])
)

invalid_ohlc_count = invalid_ohlc.sum()

df = df.loc[~invalid_ohlc].copy()

print("Invalid OHLC records removed:", invalid_ohlc_count)


# ==========================================
# 7. REGULAR TRADING HOURS FILTERING
# ==========================================

print("\n========== TRADING SESSION FILTERING ==========")

# NASDAQ regular trading session:
# 9:30 AM to 4:00 PM New York time

local_time = df["Timestamp_ET"].dt.time

start_time = pd.Timestamp("09:30").time()
end_time = pd.Timestamp("16:00").time()

weekday_mask = df["Timestamp_ET"].dt.dayofweek < 5

session_mask = (
    (local_time >= start_time) &
    (local_time < end_time)
)

before_session_filter = len(df)

df = df[weekday_mask & session_mask].copy()

session_rows_removed = before_session_filter - len(df)

print("Rows outside regular trading hours removed:",
      session_rows_removed)


# ==========================================
# 8. MISSING INTERVAL ANALYSIS
# ==========================================

print("\n========== MISSING INTERVAL ANALYSIS ==========")

# Check expected 5-minute intervals separately
# for each trading date. Do not create fake candles.

df["Trading_Date"] = df["Timestamp_ET"].dt.date

missing_intervals = 0

for date, group in df.groupby("Trading_Date"):

    timestamps = group["Timestamp_ET"].sort_values()

    expected = pd.date_range(
        start=pd.Timestamp(str(date) + " 09:30",
                           tz="America/New_York"),
        end=pd.Timestamp(str(date) + " 15:55",
                         tz="America/New_York"),
        freq="5min"
    )

    actual = pd.DatetimeIndex(timestamps)

    missing = expected.difference(actual)

    missing_intervals += len(missing)

print("Missing expected 5-minute intervals:",
      missing_intervals)


# ==========================================
# 9. FINAL CLEANING
# ==========================================

df.drop(columns=["Trading_Date"], inplace=True)

df.reset_index(drop=True, inplace=True)

final_rows = len(df)

print("\n========== FINAL DATASET ==========")

print("Initial rows:", initial_rows)
print("Final rows:", final_rows)
print("Rows removed:", initial_rows - final_rows)

print("\nFinal shape:", df.shape)

print("\nMissing values:")
print(df.isnull().sum())

print("\nDataset information:")
print(df.info())

print("\nStatistical summary:")
print(df.describe())


# ==========================================
# 10. SAVE CLEANED DATASET
# ==========================================

df.to_csv(CLEANED_FILE, index=False)

print("\nCleaned dataset saved at:")
print(CLEANED_FILE)


# ==========================================
# 11. GENERATE PREPROCESSING REPORT
# ==========================================

with open(REPORT_FILE, "w", encoding="utf-8") as f:

    f.write("NVDA STOCK DATA PREPROCESSING REPORT\n")
    f.write("=" * 45 + "\n\n")

    f.write(f"Initial rows: {initial_rows}\n")
    f.write(f"Final rows: {final_rows}\n")
    f.write(f"Rows removed: {initial_rows - final_rows}\n")

    f.write(
        f"Invalid timestamps removed: {invalid_timestamps}\n"
    )

    f.write(f"Duplicate records removed: {duplicates}\n")

    f.write(
        f"Missing essential values removed: {missing_essential}\n"
    )

    f.write(
        f"Negative volume records: {negative_volume}\n"
    )

    f.write(
        f"Invalid OHLC records removed: {invalid_ohlc_count}\n"
    )

    f.write(
        f"Rows removed by session filter: {session_rows_removed}\n"
    )

    f.write(
        f"Missing expected 5-minute intervals: {missing_intervals}\n"
    )

    f.write(f"\nFinal dataset shape: {df.shape}\n")

    f.write("\nRemaining missing values:\n")
    f.write(df.isnull().sum().to_string())

    f.write("\n\nStatistical summary:\n")
    f.write(df.describe().to_string())

print("\nPreprocessing report saved at:")
print(REPORT_FILE)

print("\n========== PREPROCESSING COMPLETED ==========")
