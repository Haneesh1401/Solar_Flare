import pandas as pd
import matplotlib.pyplot as plt
import re

# Load the dataset
df = pd.read_csv('nasa_flare_events_2010_2025.csv')

# Regular expression to parse sourceLocation (e.g., 'S25W03')
location_pattern = re.compile(r'([NESW])(\d+)([NESW])(\d+)')

# Function to convert solar coordinates to a numerical format for plotting
def parse_location(location):
    match = location_pattern.match(location)
    if not match:
        return None, None
    ns_dir, ns_deg, ew_dir, ew_deg = match.groups()
    
    # Convert degrees to a numerical value, with N/S and E/W as positive/negative
    lat = int(ns_deg) if ns_dir == 'N' else -int(ns_deg)
    lon = int(ew_deg) if ew_dir == 'W' else -int(ew_deg)
    return lat, lon

# Apply the parsing function to the DataFrame
df[['latitude', 'longitude']] = df['sourceLocation'].apply(lambda x: pd.Series(parse_location(x)))

# Drop rows where parsing failed
df.dropna(subset=['latitude', 'longitude'], inplace=True)

# Create a scatter plot of flare locations
plt.figure(figsize=(10, 6))
plt.scatter(df['longitude'], df['latitude'], alpha=0.5, s=10)
plt.title('Geographical Distribution of Solar Flare Locations')
plt.xlabel('Solar Longitude (West-East)')
plt.ylabel('Solar Latitude (North-South)')
plt.grid(True)
plt.axhline(0, color='black', linestyle='--', linewidth=0.5)
plt.axvline(0, color='black', linestyle='--', linewidth=0.5)
plt.savefig('flare_location_distribution.png')