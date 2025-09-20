from pathlib import Path
import pandas as pd
import joblib
import requests

project_root = Path(__file__).resolve().parent.parent
model_path = project_root / "models" / "model_rf.joblib"

# Load model
model = joblib.load(model_path)

# Fetch latest GOES data (last 1 day)
url = "https://services.swpc.noaa.gov/json/goes/primary/xrays-1-day.json"
data = requests.get(url).json()

df = pd.DataFrame(data)
df["time_tag"] = pd.to_datetime(df["time_tag"])
df.set_index("time_tag", inplace=True)
df = df[["flux"]].resample("10T").mean().dropna()

# Predict for latest point
latest_flux = df.iloc[-1:][["flux"]]
prediction = model.predict(latest_flux)[0]

if prediction == 1:
    print(f"⚠ High risk of solar flare in next 6 hours (flux={latest_flux.values[0][0]:.2e})")
else:
    print(f"✅ Low risk of solar flare in next 6 hours (flux={latest_flux.values[0][0]:.2e})")
