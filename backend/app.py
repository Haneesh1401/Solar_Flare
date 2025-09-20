from pathlib import Path
import joblib
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import requests
from datetime import datetime

app = FastAPI()

# Model path: go up from backend to SOLAR_FLARE_NEW, then into models
model_path = Path(__file__).resolve().parent.parent / "models" / "model_rf_improved.joblib"

print("Model path:", model_path)  # Debug print to check path

# Load model
model = joblib.load(model_path)

class FlareInput(BaseModel):
    flux: float
    month: int
    day: int

@app.post("/predict")
def predict_flare(data: FlareInput):
    input_vector = [[data.flux, data.month, data.day]]
    pred_class_num = model.predict(input_vector)[0]

    class_map_rev = {
        0: 'No Flare',
        1: 'B',
        2: 'C',
        3: 'M',
        4: 'X'
    }
    flare_class = class_map_rev.get(pred_class_num, "Unknown")

    return {"predicted_flare_class": flare_class}

@app.get("/predict_realtime")
def predict_realtime():
    try:
        # Fetch latest GOES data
        url = "https://services.swpc.noaa.gov/json/goes/primary/xrays-1-day.json"
        response = requests.get(url)
        if response.status_code != 200:
            raise Exception(f"API request failed with status {response.status_code}")
        try:
            data = response.json()
        except ValueError as e:
            raise Exception(f"Invalid JSON response: {str(e)} - Response length: {len(response.text)}")

        df = pd.DataFrame(data)
        df["time_tag"] = pd.to_datetime(df["time_tag"])
        df.set_index("time_tag", inplace=True)
        df = df[["flux"]].resample("10min").mean().dropna()

        # Get latest data point
        latest_data = df.iloc[-1]
        latest_flux = latest_data["flux"]
        current_time = latest_data.name

        # Extract features
        month = current_time.month
        day = current_time.day
        input_vector = [[latest_flux, month, day]]

        # Predict
        pred_class_num = model.predict(input_vector)[0]
        prediction_proba = model.predict_proba(input_vector)[0]

        class_map_rev = {
            0: 'No Flare',
            1: 'B',
            2: 'C',
            3: 'M',
            4: 'X'
        }
        flare_class = class_map_rev.get(pred_class_num, "Unknown")

        return {
            "time": str(current_time),
            "flux": latest_flux,
            "predicted_flare_class": flare_class,
            "probabilities": {class_map_rev.get(i, f"Class {i}"): prob for i, prob in enumerate(prediction_proba)}
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/status")
def get_status():
    return {"status": "Solar Flare Prediction API is running"}
