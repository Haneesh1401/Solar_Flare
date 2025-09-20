import os
import requests
import pandas as pd
from datetime import datetime, timedelta

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(DATA_DIR, exist_ok=True)

def fetch_goes_flux_json(date, satellite=16):
    """
    Fetch GOES X-ray flux (1-minute data) from NOAA SWPC JSON service.
    Example date: '2010-01-01'
    """
    date_str = date.strftime("%Y%m%d")
    url = f"https://services.swpc.noaa.gov/json/goes/{satellite}/xrays-1m/{date_str}_xrays-1m.json"
    
    response = requests.get(url)
    if response.status_code != 200:
        print(f"❌ No data for {date_str}")
        return None
    
    df = pd.DataFrame(response.json())
    save_path = os.path.join(DATA_DIR, f"goes_flux_{date_str}.csv")
    df.to_csv(save_path, index=False)
    print(f"✅ Saved flux data for {date_str}")
    return save_path


def fetch_flux_range(start_year=2010, end_year=2010, satellite=16):
    """
    Download daily flux data for multiple years.
    WARNING: This is many files (~365 per year)!
    """
    start_date = datetime(start_year, 1, 1)
    end_date = datetime(end_year, 12, 31)
    current = start_date
    
    while current <= end_date:
        fetch_goes_flux_json(current, satellite=satellite)
        current += timedelta(days=1)


if __name__ == "__main__":
    # Example: fetch data for Jan 2010 only
    fetch_flux_range(2010, 2010, satellite=16)
