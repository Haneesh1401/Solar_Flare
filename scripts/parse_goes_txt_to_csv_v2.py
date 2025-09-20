import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

# Project paths
project_dir = Path(__file__).resolve().parent.parent
data_dir = project_dir / "data"

years = range(2010, 2016)
all_data = []

def parse_goes_line(line):
    parts = line.strip().split()
    if len(parts) < 8:
        return None  # incomplete line
    
    datecode = parts[0]  # e.g. '31777100101'
    start_time_str = parts[1]  # e.g. '1202'
    peak_time_str = parts[2]   # e.g. '1218'
    end_time_str = parts[3]    # e.g. '1209'

    # Skip lines with invalid time strings like '//'
    if '//' in start_time_str or '//' in peak_time_str or '//' in end_time_str:
        return None

    flare_class = parts[5]     # e.g. 'B'
    flux_str = parts[7]        # e.g. '1.4E-04'

    # Extract year and day from datecode:
    year = 2000 + int(datecode[1:3])
    day_of_year = int(datecode[3:6])

    # Convert day_of_year to date
    date = datetime(year, 1, 1) + timedelta(days=day_of_year - 1)

    def parse_time(tstr):
        if len(tstr) != 4:
            return None
        try:
            hour = int(tstr[:2])
            minute = int(tstr[2:])
            return hour, minute
        except ValueError:
            return None

    start_hm = parse_time(start_time_str)
    peak_hm = parse_time(peak_time_str)
    end_hm = parse_time(end_time_str)

    if None in (start_hm, peak_hm, end_hm):
        return None

    start_dt = date.replace(hour=start_hm[0], minute=start_hm[1])
    peak_dt = date.replace(hour=peak_hm[0], minute=peak_hm[1])
    end_dt = date.replace(hour=end_hm[0], minute=end_hm[1])

    try:
        flux = float(flux_str)
    except:
        flux = None

    return {
        "start": start_dt,
        "peak": peak_dt,
        "end": end_dt,
        "flare_class": flare_class,
        "flux": flux
    }

for year in years:
    file = data_dir / f"goes_xrs_{year}.txt"
    if not file.exists():
        print(f"⚠️ File not found, skipping: {file}")
        continue

    print(f"Parsing {file}...")
    with open(file, "r") as f:
        lines = f.readlines()

    for line in lines:
        parsed = parse_goes_line(line)
        if parsed:
            all_data.append(parsed)

print(f"\nTotal flare events parsed: {len(all_data)}")

if len(all_data) == 0:
    print("❌ No valid data found. Please check your input files.")
else:
    df = pd.DataFrame(all_data)
    df.sort_values("start", inplace=True)

    csv_out = data_dir / "historical_goes_2010_2015_parsed.csv"
    df.to_csv(csv_out, index=False)

    print(f"\nParsed flare data saved to {csv_out}")
