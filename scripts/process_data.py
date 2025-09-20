from pathlib import Path
import pandas as pd
import json
from datetime import timedelta

project_root = Path(__file__).resolve().parent.parent
data_dir = project_root / "data"

# Load 10-min GOES data
goes_df = pd.read_csv(data_dir / "goes_xrs_10min.csv", parse_dates=["time_tag"], index_col="time_tag")

# Load DONKI flare events
with open(data_dir / "donki_flare.json") as f:
    flare_data = json.load(f)

flare_df = pd.DataFrame([{
    "beginTime": pd.to_datetime(f["beginTime"]),
    "classType": f.get("classType", None)
} for f in flare_data])

# Label: flare within next 6 hours = 1
labels = []
for ts in goes_df.index:
    window_end = ts + timedelta(hours=6)
    flare_in_window = flare_df[
        (flare_df["beginTime"] > ts) & (flare_df["beginTime"] <= window_end)
    ]
    labels.append(1 if not flare_in_window.empty else 0)

goes_df["flare_within_6h"] = labels

csv_path = data_dir / "labeled_dataset.csv"
goes_df.to_csv(csv_path)
print(f"Saved labeled dataset: {csv_path}")
