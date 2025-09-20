from pathlib import Path
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

# Paths
data_path = Path(__file__).resolve().parent.parent / "data" / "historical_goes_2010_2015_parsed.csv"
model_save_path = Path(__file__).resolve().parent.parent / "models" / "model_rf_improved.joblib"

# Load data
df = pd.read_csv(data_path)

# Clean data
df = df.dropna(subset=['flare_class', 'flux', 'start'])
df['flux'] = pd.to_numeric(df['flux'], errors='coerce')
df = df.dropna(subset=['flux'])
df['start'] = pd.to_datetime(df['start'], errors='coerce')
df = df.dropna(subset=['start'])
df['month'] = df['start'].dt.month
df['day'] = df['start'].dt.day

# Map flare classes
string_class_map = {'NO FLARE': 0, 'B': 1, 'C': 2, 'M': 3, 'X': 4}

def convert_flare_class(x):
    try:
        val = float(x)
        if val < 10:
            return 0
        elif val < 20:
            return 1
        elif val < 40:
            return 2
        elif val < 60:
            return 3
        else:
            return 4
    except:
        return string_class_map.get(str(x).upper(), 0)

df['flare_class_num'] = df['flare_class'].apply(convert_flare_class)

# Features and target
X = df[['flux', 'month', 'day']]
y = df['flare_class_num']

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# Save model
joblib.dump(model, model_save_path)
print(f"Model saved to {model_save_path}")
