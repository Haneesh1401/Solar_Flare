from pathlib import Path
import requests, json
from datetime import date

API_KEY = "YOUR_NASA_API_KEY"  # Replace with your NASA API key

project_root = Path(__file__).resolve().parent.parent
data_dir = project_root / "data"
data_dir.mkdir(parents=True, exist_ok=True)

start_date = date(2010, 1, 1).isoformat()
end_date = date.today().isoformat()

url = f"https://api.nasa.gov/DONKI/FLR?startDate={start_date}&endDate={end_date}&api_key={API_KEY}"
resp = requests.get(url, timeout=60)
resp.raise_for_status()
data = resp.json()

json_path = data_dir / "donki_flare.json"
with open(json_path, "w") as f:
    json.dump(data, f)
print(f"Saved DONKI flare events: {json_path}")
