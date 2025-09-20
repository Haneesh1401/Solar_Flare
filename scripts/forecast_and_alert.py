import pandas as pd
from prophet import Prophet
from sklearn.metrics import accuracy_score, precision_score
import smtplib
from email.mime.text import MIMEText
from pathlib import Path

# ==============================
# Load historical solar flare data
# ==============================
data_path = Path(__file__).resolve().parent.parent / "data" / "historical_goes_2010_2015_parsed.csv"
df = pd.read_csv(data_path)

# Prepare data for Prophet
df['start'] = pd.to_datetime(df['start'], errors='coerce')
df = df.dropna(subset=['start', 'flux'])
df = df.rename(columns={'start': 'ds', 'flux': 'y'})

# Train-test split
train = df.iloc[:-365].copy()
test = df.iloc[-365:].copy()

# Fit Prophet model
model = Prophet()
model.fit(train)

# Forecast on test dates
future = test[['ds']].copy()
forecast = model.predict(future)

# Binary classification for evaluation
threshold = 1e-6
test.loc[:, 'y_binary'] = (test['y'] > threshold).astype(int)
forecast.loc[:, 'yhat_binary'] = (forecast['yhat'] > threshold).astype(int)

# Evaluate model
accuracy = accuracy_score(test['y_binary'], forecast['yhat_binary'])
precision = precision_score(test['y_binary'], forecast['yhat_binary'])
print(f"Forecasting Accuracy: {accuracy:.4f}")
print(f"Forecasting Precision: {precision:.4f}")

# ==============================
# Email Alert System
# ==============================
def send_alert(date, predicted_flux, flare_risk, sender, receiver, password):
    """
    Send email alert with solar flare prediction
    """
    subject = f"Solar Flare Forecast for {date.strftime('%Y-%m-%d')}"
    body = f"⚠️ High solar flare risk predicted!\n\n" \
           f"Predicted flux: {predicted_flux:.2e}\n" \
           f"Date: {date.strftime('%Y-%m-%d')}\n\n" \
           f"Take necessary precautions."

    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = sender
    msg['To'] = receiver

    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(sender, password)  # App password recommended
            server.sendmail(sender, receiver, msg.as_string())
        print(f"✅ Email sent for {date.strftime('%Y-%m-%d')}")
    except Exception as e:
        print(f"❌ Failed to send email: {e}")

# ==============================
# Forecast and Alerts (High-Risk Only)
# ==============================
if __name__ == "__main__":
    # Your Gmail credentials (⚠️ Use App Passwords, not main password)
    sender = "haneesh.bhh@gmail.com"
    receiver = "shriyarao2024@gmail.com"
    email_password = "qveq gzgo ouec buuz"  # <-- Replace with your App Password

    high_risk_threshold = 5e-6
    sent_dates = set()

    # Send alerts only for high-risk forecasted days (one email per day)
    for _, row in forecast.iterrows():
        flare_risk = row['yhat'] > high_risk_threshold
        date_str = row['ds'].strftime('%Y-%m-%d')
        if flare_risk and date_str not in sent_dates:
            send_alert(row['ds'], row['yhat'], flare_risk, sender, receiver, email_password)
            sent_dates.add(date_str)

    if not sent_dates:
        print("✅ No high-risk solar flare predicted. No emails sent.")

    print("🌞 Solar flare forecasting and high-risk email alerts completed.")
