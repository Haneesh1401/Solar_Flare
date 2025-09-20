import pandas as pd
from prophet import Prophet
import joblib
from pathlib import Path
from datetime import datetime
import numpy as np

def load_and_train_prophet(data_path):
    df = pd.read_csv(data_path)
    df['start'] = pd.to_datetime(df['start'], errors='coerce')
    df = df.dropna(subset=['start', 'flux'])
    df = df.rename(columns={'start': 'ds', 'flux': 'y'})

    model = Prophet()
    model.fit(df)
    return model

def predict_flux_for_date(model, date):
    future = pd.DataFrame({'ds': [date]})
    forecast = model.predict(future)
    return forecast.iloc[0]['yhat']

def load_rf_model(model_path):
    return joblib.load(model_path)

def predict_flare_class(rf_model, flux, month, day):
    # Predict flare class using RF model
    features = pd.DataFrame([[flux, month, day]], columns=['flux', 'month', 'day'])
    prediction = rf_model.predict(features)[0]
    flare_classes = {0: 'NO FLARE', 1: 'B', 2: 'C', 3: 'M', 4: 'X'}
    return flare_classes.get(prediction, 'UNKNOWN')

def assess_danger_level(flare_class):
    levels = {
        'NO FLARE': 'No danger',
        'B': 'Low danger',
        'C': 'Moderate danger',
        'M': 'High danger',
        'X': 'Extreme danger'
    }
    return levels.get(flare_class, 'Unknown danger level')

def multiple_instance_learning_ensemble(rf_model, flux, month, day, n_instances=5, noise_level=0.05):
    # Generate multiple noisy instances around the flux value to simulate MIL
    predictions = []
    for _ in range(n_instances):
        noisy_flux = flux * (1 + np.random.uniform(-noise_level, noise_level))
        features = pd.DataFrame([[noisy_flux, month, day]], columns=['flux', 'month', 'day'])
        pred = rf_model.predict(features)[0]
        predictions.append(pred)
    # Majority vote
    pred_counts = pd.Series(predictions).value_counts()
    majority_pred = pred_counts.idxmax()
    flare_classes = {0: 'NO FLARE', 1: 'B', 2: 'C', 3: 'M', 4: 'X'}
    return flare_classes.get(majority_pred, 'UNKNOWN')

def main():
    data_path = Path(__file__).resolve().parent.parent / "data" / "historical_goes_2010_2015_parsed.csv"
    model_path = Path(__file__).resolve().parent.parent / "models" / "model_rf_improved.joblib"

    # Load and train Prophet model
    print("Training Prophet model on historical data...")
    prophet_model = load_and_train_prophet(data_path)

    # Load RandomForest model
    print("Loading RandomForest model...")
    rf_model = load_rf_model(model_path)

    # Get user input date
    user_input = input("Enter date to predict solar flare (YYYY-MM-DD): ")
    try:
        date = datetime.strptime(user_input, "%Y-%m-%d")
    except ValueError:
        print("Invalid date format. Please use YYYY-MM-DD.")
        return

    # Predict flux for the date
    predicted_flux = predict_flux_for_date(prophet_model, date)
    print(f"Predicted solar flare flux on {date.date()}: {predicted_flux:.2e}")

    # Extract month and day for RF model features
    month = date.month
    day = date.day

    # Predict flare class using MIL ensemble
    predicted_flare_class = multiple_instance_learning_ensemble(rf_model, predicted_flux, month, day)
    danger_level = assess_danger_level(predicted_flare_class)

    print(f"Predicted solar flare class: {predicted_flare_class}")
    print(f"Danger level: {danger_level}")

if __name__ == "__main__":
    main()
