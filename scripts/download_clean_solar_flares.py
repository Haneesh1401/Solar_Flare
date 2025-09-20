import os
import requests
import pandas as pd
from datetime import datetime

# Define the base URL for NOAA SWPC FTP archive
base_url = "ftp://ftp.swpc.noaa.gov/pub/warehouse/"

# Define the years for which data is needed
years = [str(year) for year in range(2015, 2026)]

# Initialize an empty list to store the cleaned data
data = []

# Loop through each year to download and process the data
for year in years:
    file_url = f"{base_url}xray-flux-{year}.txt"
    try:
        # Download the file
        response = requests.get(file_url)
        response.raise_for_status()  # Check if the request was successful
        lines = response.text.splitlines()
        
        # Process each line in the file
        for line in lines:
            if line.startswith("Date"):  # Skip header lines
                continue
            parts = line.split()
            if len(parts) < 7:
                continue  # Skip lines with insufficient data
            date_str = parts[0]
            time_str = parts[1]
            flare_class = parts[2]
            xray_flux = parts[3]
            # Convert date and time to a single datetime object
            try:
                timestamp = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M:%S")
            except ValueError:
                continue  # Skip lines with invalid date/time
            # Append the data to the list
            data.append([timestamp, flare_class, xray_flux])
    except requests.exceptions.RequestException as e:
        print(f"Failed to download or process data for {year}: {e}")

# Create a DataFrame from the collected data
df = pd.DataFrame(data, columns=["timestamp", "flare_class", "xray_flux"])

# Ensure the data folder exists
output_folder = "data"
os.makedirs(output_folder, exist_ok=True)

# Save the cleaned data to a CSV file inside the data folder
output_path = os.path.join(output_folder, "solar_flares_cleaned.csv")
df.to_csv(output_path, index=False)

print(f"Data has been successfully downloaded and cleaned.\nSaved at: {output_path}")
