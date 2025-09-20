import streamlit as st
import pandas as pd
import joblib
from datetime import datetime
from pathlib import Path
import sys
from prophet import Prophet
import matplotlib.pyplot as plt
import smtplib
from email.mime.text import MIMEText

# ==============================
# Set paths
# ==============================
project_root = Path(__file__).resolve().parent.parent
model_path = project_root / "models" / "model_rf_improved.joblib"
data_path = project_root / "data" / "historical_goes_2010_2015_parsed.csv"

# Debug: check if files exist
if not model_path.exists():
    st.error(f"Model file not found: {model_path}")
    sys.exit(1)
if not data_path.exists():
    st.error(f"Data file not found: {data_path}")
    sys.exit(1)

# ==============================
# Load Random Forest model
# ==============================
@st.cache_resource
def load_rf_model():
    return joblib.load(model_path)

rf_model = load_rf_model()

# ==============================
# Load historical data for Prophet
# ==============================
df = pd.read_csv(data_path)
df['start'] = pd.to_datetime(df['start'], errors='coerce')
df = df.dropna(subset=['start', 'flux'])
df_prophet = df.rename(columns={'start': 'ds', 'flux': 'y'})

# Train Prophet model
@st.cache_resource
def load_prophet_model():
    model = Prophet()
    model.fit(df_prophet)
    return model

prophet_model = load_prophet_model()

# ==============================
# Email Alert Function
# ==============================
def send_alert_email(date, predicted_flux, flare_class, sender, receiver, password):
    subject = f"Solar Flare Alert for {date.strftime('%Y-%m-%d')}"
    body = f"⚠️ High solar flare predicted!\n\n" \
           f"Predicted Flux: {predicted_flux:.2e} W/m²\n" \
           f"Flare Class: {flare_class}\n" \
           f"Date: {date.strftime('%Y-%m-%d')}\n\n" \
           f"Take necessary precautions."

    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = sender
    msg['To'] = receiver

    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(sender, password)
            server.sendmail(sender, receiver, msg.as_string())
        st.success(f"✅ Alert email sent to {receiver}")
    except Exception as e:
        st.error(f"❌ Failed to send email: {e}")

# ==============================
# UI
# ==============================
st.title("☀️ Solar Flare Future Risk Prediction")
st.write("Pick a future date, and the app will estimate the solar flare risk for that day.")

selected_date = st.date_input("Select a future date", min_value=datetime.today())

if st.button("Predict Risk"):
    try:
        # Predict flux using Prophet
        future = pd.DataFrame({'ds': [selected_date]})
        forecast = prophet_model.predict(future)
        predicted_flux = forecast.iloc[0]['yhat']

        if predicted_flux < 0:
            st.warning("Predicted flux is negative, which is unrealistic. Please try another date.")
        else:
            month = selected_date.month
            day = selected_date.day

            # Predict flare class using RF model
            input_df = pd.DataFrame([[predicted_flux, month, day]], columns=['flux', 'month', 'day'])
            pred_class_num = rf_model.predict(input_df)[0]

            class_map_rev = {
                0: 'NO FLARE',
                1: 'B',
                2: 'C',
                3: 'M',
                4: 'X'
            }
            alert_levels = {
                'NO FLARE': 'No danger',
                'B': 'Low danger',
                'C': 'Moderate danger',
                'M': 'High danger',
                'X': 'Extreme danger'
            }

            flare_class = class_map_rev.get(pred_class_num, "Unknown")
            alert = alert_levels.get(flare_class, "No Alert")

            # Calculate prediction probabilities
            pred_proba = rf_model.predict_proba(input_df)[0]
            proba_dict = {class_map_rev[i]: prob for i, prob in enumerate(pred_proba)}

            st.subheader(f"Predicted Solar Flare Flux: {predicted_flux:.2e} W/m²")
            st.subheader(f"Predicted Flare Class: {flare_class}")
            st.subheader(f"Danger Level: {alert}")

            st.subheader("Prediction Probabilities:")
            for cls, prob in proba_dict.items():
                st.write(f"{cls}: {prob:.4f}")

            st.caption(f"(Prediction for {selected_date})")

            # Plot historical flux and predicted flux
            fig, ax = plt.subplots()
            ax.plot(df['start'], df['flux'], label='Historical Flux')
            ax.scatter([selected_date], [predicted_flux], color='red', label='Predicted Flux')
            ax.set_xlabel('Date')
            ax.set_ylabel('Flux (W/m²)')
            ax.set_title('Solar Flare Flux Prediction')
            ax.legend()
            st.pyplot(fig)

            # ==============================
            # Email Alert (for high-risk flares only)
            # ==============================
            sender = st.secrets["email"]["sender"]
            receiver = st.secrets["email"]["receiver"]
            email_password = st.secrets["email"]["app_password"]

            if flare_class in ['M', 'X']:
                send_alert_email(selected_date, predicted_flux, flare_class, sender, receiver, email_password)

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
