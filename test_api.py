
import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("MASSIVE_API_KEY")

if not API_KEY:
    raise ValueError("API key not found in .env file!")

url = "https://api.massive.com/v1/marketstatus/now"

response = requests.get(
    url,
    params={"apiKey": API_KEY},
    timeout=30
)

print("Status code:", response.status_code)

if response.status_code == 200:
    print("API connection successful!")
    print(response.json())
else:
    print("API request failed:")
    print(response.text)
