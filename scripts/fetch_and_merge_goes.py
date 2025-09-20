import requests
import pandas as pd

API_KEY = "DEMO_KEY"  # Replace with your NASA API key

def fetch_nasa_flare_events(start_date="2010-01-01", end_date="2025-12-31"):
    url = "https://api.nasa.gov/DONKI/FLR"
    params = {"startDate": start_date, "endDate": end_date, "api_key": API_KEY}
    resp = requests.get(url, params=params)
    resp.raise_for_status()
    data = resp.json()
    df = pd.json_normalize(data)
    df.to_csv("nasa_flare_events_2010_2025.csv", index=False)
    print(f"Saved {len(df)} flare events to nasa_flare_events_2010_2025.csv")
    return df

if __name__ == "__main__":
    df = fetch_nasa_flare_events()
    print(df.head())
