from pathlib import Path
import pandas as pd
import joblib
import requests
from datetime import datetime
import time
import smtplib
from email.mime.text import MIMEText

# Paths
project_root = Path(__file__).resolve().parent.parent
model_path = project_root / "models" / "model_rf_improved.joblib"

# Load the improved model
model = joblib.load(model_path)
print("Loaded improved RandomForest model")

# Alert function
def send_alert(date, predicted_class, flux):
    sender = "your_email@example.com"
    receiver = "alert_recipient@example.com"
    subject = f"Solar Flare Alert: {predicted_class} class predicted"
    body = f"Predicted flare class: {predicted_class}\nFlux: {flux:.2e}\nTime: {date}"
    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = sender
    msg['To'] = receiver

    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(sender, "your_email_password")
            server.sendmail(sender, receiver, msg.as_string())
        print(f"Alert sent for {predicted_class} flare at {date}")
    except Exception as e:
        print(f"Failed to send alert: {e}")

# Real-time prediction loop
while True:
    try:
        # Fetch latest GOES data (last 1 day)
        url = "https://services.swpc.noaa.gov/json/goes/primary/xrays-1-day.json"
        data = requests.get(url).json()

        df = pd.DataFrame(data)
        df["time_tag"] = pd.to_datetime(df["time_tag"])
        df.set_index("time_tag", inplace=True)
        df = df[["flux"]].resample("10min").mean().dropna()

        # Get latest data point
        latest_data = df.iloc[-1]
        latest_flux = latest_data["flux"]
        current_time = latest_data.name

        # Extract features for prediction (flux, month, day)
        month = current_time.month
        day = current_time.day
        features = pd.DataFrame([[latest_flux, month, day]], columns=['flux', 'month', 'day'])

        # Make prediction
        prediction = model.predict(features)[0]
        prediction_proba = model.predict_proba(features)[0]

        # Map prediction back to flare class
        flare_classes = {0: 'NO FLARE', 1: 'B', 2: 'C', 3: 'M', 4: 'X'}
        predicted_class = flare_classes.get(prediction, 'UNKNOWN')

        print(f"Latest GOES data at {current_time}")
        print(f"Flux: {latest_flux:.2e}")
        print(f"Predicted flare class: {predicted_class}")
        print(f"Prediction probabilities:")
        for i, prob in enumerate(prediction_proba):
            print(f"  {flare_classes.get(i, f'Class {i}')}: {prob:.4f}")

        # Risk assessment
        if prediction >= 3:  # M or X class
            print("⚠ HIGH RISK: Major solar flare predicted")
            send_alert(current_time, predicted_class, latest_flux)
        elif prediction >= 2:  # C class
            print("⚠ MEDIUM RISK: Moderate solar flare predicted")
        elif prediction >= 1:  # B class
            print("⚠ LOW RISK: Minor solar flare predicted")
        else:
            print("✅ LOW RISK: No significant flare predicted")

    except Exception as e:
        print(f"Error during prediction: {e}")

    # Wait 10 minutes before next prediction
    time.sleep(600)
