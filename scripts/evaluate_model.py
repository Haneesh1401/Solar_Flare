from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report

print("Starting evaluation script...")

# Paths relative to this script
data_path = Path(__file__).resolve().parent.parent / "data" / "historical_goes_2010_2015_parsed.csv"
model_path = Path(__file__).resolve().parent.parent / "models" / "model_rf_improved.joblib"  # <-- update if different

print(f"Data file: {data_path}")
print(f"Model file: {model_path}")

# Load data
df = pd.read_csv(data_path)
print(f"Loaded data with {len(df)} rows")

# Drop rows with missing flare_class, flux, or start date
df = df.dropna(subset=['flare_class', 'flux', 'start'])

# Convert flux to numeric, coerce errors and drop missing
df['flux'] = pd.to_numeric(df['flux'], errors='coerce')
df = df.dropna(subset=['flux'])

# Parse 'start' datetime and extract month and day
df['start'] = pd.to_datetime(df['start'], errors='coerce')
df = df.dropna(subset=['start'])
df['month'] = df['start'].dt.month
df['day'] = df['start'].dt.day

# Map string flare classes to numbers
string_class_map = {'NO FLARE': 0, 'B': 1, 'C': 2, 'M': 3, 'X': 4}

def convert_flare_class(x):
    try:
        # Try to convert to float, then map using thresholds
        val = float(x)
        if val < 10:
            return 0  # No Flare proxy
        elif val < 20:
            return 1  # B proxy
        elif val < 40:
            return 2  # C proxy
        elif val < 60:
            return 3  # M proxy
        else:
            return 4  # X proxy
    except:
        # If string, map using dict (case insensitive)
        x_str = str(x).upper()
        return string_class_map.get(x_str, 0)  # Default to No Flare if unknown

# Apply conversion
df['flare_class_num'] = df['flare_class'].apply(convert_flare_class)

print("Converted flare classes:")
print(df['flare_class_num'].value_counts())

# Features and target
X = df[['flux', 'month', 'day']]
y = df['flare_class_num']

print(f"Features shape: {X.shape}, Target shape: {y.shape}")

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load your trained model
model = joblib.load(model_path)
print("Model loaded successfully")

# Predict on test set
y_pred = model.predict(X_test)
print("Prediction completed")

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
conf_mat = confusion_matrix(y_test, y_pred)

# Print results
print("\nModel Evaluation Metrics:")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")

print("\nConfusion Matrix:")
print(conf_mat)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

print("Script finished")
