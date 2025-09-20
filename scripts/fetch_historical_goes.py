import requests
from pathlib import Path

# Path to the "data" folder
project_dir = Path(__file__).resolve().parent.parent
data_dir = project_dir / "data"
data_dir.mkdir(exist_ok=True)

# Years of data you want to fetch
start_year = 2010
end_year = 2024

for year in range(start_year, end_year + 1):
    url = f"https://www.ngdc.noaa.gov/stp/space-weather/solar-data/solar-features/solar-flares/x-rays/goes/xrs/goes-xrs-report_{year}.txt"
    print(f"Downloading {year} data from {url} ...")

    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"❌ Failed to download {year}: {e}")
        continue

    # Save file in data folder
    out_file = data_dir / f"goes_xrs_{year}.txt"
    with open(out_file, "w") as f:
        f.write(resp.text)

    print(f"✅ Saved {year} to {out_file}")

print("\nAll downloads complete.")
