from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

project_root = Path(__file__).resolve().parent.parent
data_dir = project_root / "data"
model_dir = project_root / "models"
model_dir.mkdir(parents=True, exist_ok=True)

# Load your parsed GOES data (adjust filename if needed)
df = pd.read_csv(data_dir / "historical_goes_2010_2015_parsed.csv")

# Map flare classes to numeric labels for classification
flare_class_map = {'B': 1, 'C': 2, 'M': 3, 'X': 4}
df = df[df['flare_class'].isin(flare_class_map.keys())]  # filter only known classes
df['flare_class_num'] = df['flare_class'].map(flare_class_map)

# Features (flux) and target (flare class)
X = df[['flux']]
y = df['flare_class_num']

# Split dataset (shuffle is True by default, you can set False if needed)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
preds = model.predict(X_test)
print(classification_report(y_test, preds))

# Save the trained model
model_path = model_dir / "model_rf.joblib"
joblib.dump(model, model_path)
print(f"Model saved to {model_path}")
