import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.svm import SVC
import joblib
from pathlib import Path

def main():
    data_path = Path(__file__).resolve().parent / "historical_goes_2010_2025_cleaned.csv"
    df = pd.read_csv(data_path)

    df = df.dropna(subset=['classType', 'classNumeric', 'beginTime'])
    df['flux'] = pd.to_numeric(df['classNumeric'], errors='coerce')
    df = df.dropna(subset=['flux'])
    df['start'] = pd.to_datetime(df['beginTime'], errors='coerce')
    df = df.dropna(subset=['start'])
    df['peak'] = pd.to_datetime(df['peakTime'], errors='coerce')
    df['end'] = pd.to_datetime(df['endTime'], errors='coerce')
    df = df.dropna(subset=['peak', 'end'])

    # Feature engineering
    df['month'] = df['start'].dt.month
    df['day'] = df['start'].dt.day
    df['year'] = df['start'].dt.year
    df['duration_minutes'] = (df['end'] - df['start']).dt.total_seconds() / 60.0
    df['rise_minutes'] = (df['peak'] - df['start']).dt.total_seconds() / 60.0
    df['decay_minutes'] = (df['end'] - df['peak']).dt.total_seconds() / 60.0

    def convert_flare_class(x):
        try:
            val = float(x)
            if val < 10:
                return 0
            else:
                return 1
        except:
            if str(x).upper() == 'NO FLARE':
                return 0
            else:
                return 1

    df['flare_class_num'] = df['classType'].apply(convert_flare_class)

    # For MIL, create bags: group by year-month as bags
    df['bag'] = df['year'].astype(str) + '-' + df['month'].astype(str)

    # Features
    features = ['flux', 'month', 'day', 'year', 'duration_minutes', 'rise_minutes', 'decay_minutes']

    # Aggregate features per bag (max, mean, etc.)
    bag_features = df.groupby('bag')[features].agg(['max', 'mean', 'min']).reset_index()
    bag_features.columns = ['bag'] + [f'{col}_{agg}' for col in features for agg in ['max', 'mean', 'min']]

    # Bag label: mean flare class in the bag, rounded
    bag_labels = df.groupby('bag')['flare_class_num'].mean().round().astype(int).reset_index()
    bag_labels.columns = ['bag', 'flare_class_num']

    # Merge
    bag_df = pd.merge(bag_features, bag_labels, on='bag')

    print(f"Number of bags: {len(bag_df)}")
    print(f"Unique bags: {bag_df['bag'].unique()}")

    X = bag_df.drop(['bag', 'flare_class_num'], axis=1)
    y = bag_df['flare_class_num']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Use SVM as MIL classifier (bag-level)
    model = SVC(random_state=42, C=0.1)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    print("Multiple Instance Learning (MIL) with SVM Results:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred, average='weighted'):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred, average='weighted'):.4f}")
    print(f"F1 Score: {f1_score(y_test, y_pred, average='weighted'):.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    model_save_path = Path(__file__).resolve().parent.parent / "models" / "mil_model.joblib"
    joblib.dump(model, model_save_path)
    print(f"Model saved to {model_save_path}")

if __name__ == "__main__":
    main()
