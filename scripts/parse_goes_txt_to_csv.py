import pandas as pd
from pathlib import Path
import re

project_dir = Path(__file__).resolve().parent.parent
data_dir = project_dir / "data"

years = range(2010, 2016)
all_data = []

timestamp_pattern = re.compile(r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}')

for year in years:
    file = data_dir / f"goes_xrs_{year}.txt"
    if not file.exists():
        print(f"⚠️ File not found, skipping: {file}")
        continue

    print(f"Parsing {file}...")
    with open(file, "r") as f:
        lines = f.readlines()

    data_start_index = 0
    for i, line in enumerate(lines):
        if timestamp_pattern.match(line):
            data_start_index = i
            break

    for line in lines[data_start_index:]:
        parts = line.strip().split()
        if len(parts) >= 2:
            timestamp = parts[0]
            flux = parts[1]
            if not timestamp_pattern.match(timestamp):
                continue
            try:
                flux_val = float(flux)
            except ValueError:
                flux_val = None
            if flux_val is not None:
                all_data.append({"timestamp": timestamp, "flux": flux_val})

print(f"\nTotal data rows collected: {len(all_data)}")

if len(all_data) == 0:
    print("❌ No valid data found. Please check your input files.")
else:
    df = pd.DataFrame(all_data)
    print(f"DataFrame columns: {df.columns.tolist()}")

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.sort_values("timestamp", inplace=True)

    csv_out = data_dir / "historical_goes_2010_2015.csv"
    df.to_csv(csv_out, index=False)

    print(f"\nParsed data saved to {csv_out}")
    print(f"Total rows in CSV: {len(df)}")
