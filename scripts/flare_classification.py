import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

def load_and_feature_engineer():
    """Load the historical GOES flare dataset and create features."""
    # Build path to CSV file
    csv_path = os.path.join(os.path.dirname(__file__), "..", "data", "historical_goes_2010_2015_parsed.csv")
    print(f"Loading CSV from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Initial data shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Sample data:\n{df.head()}")

    # Convert important columns to datetime
    time_columns = ["start", "peak", "end"]
    for col in time_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)
            print(f"After converting {col}: {df[col].dtype}, NaNs: {df[col].isna().sum()}")

    # Rename columns to match expected names
    df.rename(columns={"start": "beginTime", "peak": "peakTime", "end": "endTime", "flare_class": "classNumeric"}, inplace=True)
    print(f"After renaming columns: {df.columns.tolist()}")

    # Drop rows where any critical time column is missing
    initial_len = len(df)
    df = df.dropna(subset=["beginTime", "peakTime", "endTime"])
    print(f"Dropped {initial_len - len(df)} rows due to missing time columns. Remaining: {len(df)}")

    # Feature engineering: duration, rise, and decay
    df["duration_minutes"] = (df["endTime"] - df["beginTime"]).dt.total_seconds() / 60.0
    df["rise_minutes"] = (df["peakTime"] - df["beginTime"]).dt.total_seconds() / 60.0
    df["decay_minutes"] = (df["endTime"] - df["peakTime"]).dt.total_seconds() / 60.0
    print(f"Feature engineering done. Duration NaNs: {df['duration_minutes'].isna().sum()}")

    # Ensure classNumeric is numeric
    df["classNumeric"] = pd.to_numeric(df["classNumeric"], errors='coerce')
    # Map flare_class to categories: A=1, B=2, C=3, M=4, X=5 based on numeric value
    df["classNumeric"] = df["classNumeric"].apply(lambda x: 1 if x < 10 else (2 if x < 20 else (3 if x < 30 else (4 if x < 40 else 5))))
    print(f"After mapping to categories; NaNs in classNumeric: {df['classNumeric'].isna().sum()}")

    # Drop rows with missing values in engineered features or classNumeric
    df = df.dropna(subset=["duration_minutes", "rise_minutes", "decay_minutes", "classNumeric", "flux"])
    print(f"Final data shape after dropping NaNs: {df.shape}")

    return df

def prepare_dataset(df):
    """Split dataset into features and target."""
    # Features (X) and Target (y)
    X = df[["duration_minutes", "rise_minutes", "decay_minutes", "flux"]]
    y = df["classNumeric"]

    # Train-Test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale features for better ML performance
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test

def train_and_evaluate(X_train, X_test, y_train, y_test):
    """Train a RandomForestClassifier and print evaluation results."""
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

def main():
    print("🔄 Loading and processing data...")
    df = load_and_feature_engineer()
    print(f"✅ Loaded {len(df)} events after cleaning.\n")

    if len(df) == 0:
        print("❌ No valid data available after cleaning. Please check the data source.")
        return

    print("📊 Preparing dataset for ML...")
    X_train, X_test, y_train, y_test = prepare_dataset(df)

    print("🤖 Training model and evaluating performance...")
    train_and_evaluate(X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    main()
