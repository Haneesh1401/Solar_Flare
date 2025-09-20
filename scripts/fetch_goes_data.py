from pathlib import Path
import requests, json
import pandas as pd

project_root = Path(__file__).resolve().parent.parent
data_dir = project_root / "data"
data_dir.mkdir(parents=True, exist_ok=True)

# Fetch GOES XRS last 7 days
url = "https://services.swpc.noaa.gov/json/goes/primary/xrays-7-day.json"
resp = requests.get(url, timeout=30)
resp.raise_for_status()
data = resp.json()

# Save raw JSON
raw_path = data_dir / "goes_xrs_raw.json"
with open(raw_path, "w") as f:
    json.dump(data, f)
print(f"Saved raw GOES data: {raw_path}")

# Convert to DataFrame & resample to 10 minutes
df = pd.DataFrame(data)
df["time_tag"] = pd.to_datetime(df["time_tag"])
df.set_index("time_tag", inplace=True)
df = df[["flux"]].resample("10T").mean()

csv_path = data_dir / "goes_xrs_10min.csv"
df.to_csv(csv_path)
print(f"Saved 10-min GOES data: {csv_path}")
