
import os
import time
import requests
import pandas as pd
from dotenv import load_dotenv

# ==================================================
# 1. CONFIGURATION
# ==================================================

load_dotenv()

API_KEY = os.getenv("MASSIVE_API_KEY")

TICKER = "NVDA"
START_DATE = "2024-11-04"
END_DATE = "2026-10-30"

OUTPUT_DIR = "data/raw"
OUTPUT_FILE = os.path.join(
    OUTPUT_DIR, "NVDA_5min_regular_2y.csv"
)

BASE_URL = (
    f"https://api.massive.com/v2/aggs/ticker/{TICKER}"
    f"/range/5/minute/{START_DATE}/{END_DATE}"
)

if not API_KEY:
    raise ValueError("MASSIVE_API_KEY not found in .env file")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==================================================
# 2. DOWNLOAD WITH PAGINATION
# ==================================================

session = requests.Session()

params = {
    "adjusted": "true",
    "sort": "asc",
    "limit": 50000,
    "apiKey": API_KEY
}

all_results = []
url = BASE_URL
page = 1

print("Starting NVDA historical data download...")
print("Period:", START_DATE, "to", END_DATE)
print("Interval: 5 minutes")

while url:

    print(f"\nDownloading page {page}...")

    for attempt in range(5):
        response = session.get(
            url,
            params=params if page == 1 else None,
            timeout=60
        )

        if response.status_code == 429:
            wait = 15 * (attempt + 1)
            print(f"Rate limited. Waiting {wait} seconds...")
            time.sleep(wait)
            continue

        break

    if response.status_code == 429:
        raise RuntimeError("API rate limit exceeded after retries.")

    if response.status_code != 200:
        print("HTTP status:", response.status_code)
        print("API response:", response.text[:1500])
        response.raise_for_status()

    data = response.json()

    if data.get("status") == "ERROR":
        raise RuntimeError(data.get("error", "API returned an error"))

    results = data.get("results", [])

    if results:
        all_results.extend(results)
        print("Bars received:", len(results))
    else:
        print("No bars returned on this page.")

    # Follow the API pagination URL
    url = data.get("next_url")

    if url:
        if "apiKey=" not in url:
            separator = "&" if "?" in url else "?"
            url = f"{url}{separator}apiKey={API_KEY}"

        # Free plan has a 5 calls/minute limit
        time.sleep(12)

    page += 1

session.close()

if not all_results:
    raise ValueError(
        "No historical bars received. Check API access and date range."
    )

print("\nTotal downloaded bars:", len(all_results))

# ==================================================
# 3. CONVERT TO DATAFRAME
# ==================================================

df = pd.DataFrame(all_results)

rename_columns = {
    "t": "Timestamp",
    "o": "Open",
    "h": "High",
    "l": "Low",
    "c": "Close",
    "v": "Volume",
    "vw": "VWAP",
    "n": "Transactions"
}

df.rename(columns=rename_columns, inplace=True)

required = [
    "Timestamp", "Open", "High",
    "Low", "Close", "Volume"
]

missing_columns = [
    col for col in required if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"API response is missing columns: {missing_columns}"
    )

# Unix milliseconds to UTC
df["Timestamp"] = pd.to_datetime(
    df["Timestamp"],
    unit="ms",
    utc=True
)

# Convert prices and volume to numeric
numeric_columns = [
    "Open", "High", "Low", "Close", "Volume"
]

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Remove invalid records and duplicates
df = df.dropna(subset=required)
df = df.drop_duplicates(subset=["Timestamp"])
df = df.sort_values("Timestamp").reset_index(drop=True)

# ==================================================
# 4. FILTER REGULAR MARKET SESSION
# ==================================================

# Convert UTC timestamps to US Eastern time.
df["Timestamp_ET"] = (
    df["Timestamp"].dt.tz_convert("America/New_York")
)

# Keep 9:30 AM inclusive and 4:00 PM exclusive.
local_time = df["Timestamp_ET"].dt.time

start_time = pd.Timestamp("09:30").time()
end_time = pd.Timestamp("16:00").time()

df = df[
    (local_time >= start_time) &
    (local_time < end_time)
].copy()

df = df.reset_index(drop=True)

# ==================================================
# 5. DATA QUALITY CHECKS
# ==================================================

invalid = (
    (df["High"] < df["Low"]) |
    (df["High"] < df["Open"]) |
    (df["High"] < df["Close"]) |
    (df["Low"] > df["Open"]) |
    (df["Low"] > df["Close"]) |
    (df[["Open", "High", "Low", "Close"]] <= 0).any(axis=1) |
    (df["Volume"] < 0)
)

print("\nInvalid OHLCV rows:", int(invalid.sum()))

# Remove invalid OHLCV rows
df = df.loc[~invalid].copy()

# Select output columns
output_columns = [
    "Timestamp",
    "Timestamp_ET",
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "VWAP",
    "Transactions"
]

df = df[
    [col for col in output_columns if col in df.columns]
]

# ==================================================
# 6. SAVE DATASET
# ==================================================

df.to_csv(OUTPUT_FILE, index=False)

print("\n" + "=" * 55)
print("NVDA DATASET DOWNLOAD COMPLETED")
print("=" * 55)

print("Total candles:", len(df))
print("Start timestamp:", df["Timestamp"].min())
print("End timestamp:", df["Timestamp"].max())

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate timestamps:", df["Timestamp"].duplicated().sum())

print("\nFirst five rows:")
print(df.head())

print("\nLast five rows:")
print(df.tail())

print("\nSaved successfully to:", OUTPUT_FILE)
