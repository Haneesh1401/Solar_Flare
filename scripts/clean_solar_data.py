import pandas as pd

# Sample raw data (replace this with your full file)
raw_data = """
31777110101  0017 0030 0022  B 23  G15  1.4E-04 11140
31777110101  0049 0057 0053  B 22  G15  7.3E-05 11140
31777110102  0903 0922 0910  B 46  G15  4.0E-04 11140
31777110103  2326 2346 2335  C 11  G15  9.6E-04 11142
"""

# Convert raw data into list of rows
rows = [line.split() for line in raw_data.strip().split('\n')]

# Prepare cleaned data
cleaned_data = []
for r in rows:
    event_id = r[0]
    
    # Extract date from event ID (first 8 digits = YYYYMMDD)
    date_str = event_id[:8]
    date_formatted = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
    
    start_time = r[1][:2] + ":" + r[1][2:]
    peak_time  = r[2][:2] + ":" + r[2][2:]
    end_time   = r[3][:2] + ":" + r[3][2:]
    
    flare_class = r[4]
    intensity = r[7]  # flux value
    satellite = r[6]
    active_region = r[8] if len(r) > 8 else ""
    
    cleaned_data.append([date_formatted, start_time, peak_time, end_time,
                         flare_class, intensity, satellite, active_region])

# Convert to DataFrame
df = pd.DataFrame(cleaned_data, columns=[
    "date", "start_time", "peak_time", "end_time",
    "flare_class", "intensity", "satellite", "active_region"
])

# Save as CSV
df.to_csv("solar_flares_cleaned.csv", index=False)
print(df)
