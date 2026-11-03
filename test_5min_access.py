
import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("MASSIVE_API_KEY")

if not API_KEY:
    raise ValueError("MASSIVE_API_KEY not found in .env")

# One completed trading day for the access test
url = (
    "https://api.massive.com/v2/aggs/ticker/"
    "NVDA/range/5/minute/2026-10-29/2026-10-29"
)

params = {
    "adjusted": "true",
    "sort": "asc",
    "limit": 50000,
    "apiKey": API_KEY
}

response = requests.get(url, params=params, timeout=30)

print("HTTP status:", response.status_code)

if response.status_code == 200:
    data = response.json()
    results = data.get("results", [])

    print("API status:", data.get("status"))
    print("Number of 5-minute bars:", len(results))

    if results:
        print("\nFirst returned bar:")
        print(results[0])
        print("\nSUCCESS: Your account can access historical 5-minute NVDA data.")
    else:
        print("\nNo bars returned for this date. Check the date and API response.")
else:
    print("\nAPI access test failed.")
    print("Response:", response.text[:1500])

    if response.status_code in (401, 403):
        print(
            "\nThis may be an API key or subscription entitlement restriction."
        )
