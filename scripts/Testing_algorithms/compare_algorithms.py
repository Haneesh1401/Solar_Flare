import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import tensorflow as tf
from pathlib import Path
import os

def load_data():
    data_path = Path(__file__).resolve().parent.parent / "data" / "historical_goes_2010_2015_parsed.csv"
    df = pd.read_csv(data_path)

    df = df.dropna(subset=['flare_class', 'flux', 'start'])
    df['flux'] = pd.to_numeric(df['flux'], errors='coerce')
    df = df.dropna(subset=['flux'])
    df['start'] = pd.to_datetime(df['start'], errors='coerce')
    df = df.dropna(subset=['start'])
    df['month'] = df['start'].dt.month
    df['day'] = df['start'].dt.day

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

    X = df[['flux', 'month', 'day']]
    y = df['flare_class_num']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_test_scaled, y_test

def evaluate_model(model, X_test, y_test, name):
    if name == 'TensorFlow':
        y_pred_proba = model.predict(X_test)
        y_pred = np.argmax(y_pred_proba, axis=1)
    else:
        y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    return accuracy, precision, recall, f1

def main():
    X_test, y_test = load_data()
    models_dir = Path(__file__).resolve().parent.parent / "models"

    results = {}

    # Load and evaluate each model
    model_files = {
        'Random Forest': 'random_forest_model.joblib',
        'XGBoost': 'xgboost_model.joblib',
        'TensorFlow': 'tensorflow_model.h5',
        'Logistic Regression': 'logistic_regression_model.joblib',
        'SVM': 'svm_model.joblib',
        'LightGBM': 'lightgbm_model.joblib',
        'CatBoost': 'catboost_model.joblib'
    }

    for name, file in model_files.items():
        path = models_dir / file
        if os.path.exists(path):
            if name == 'TensorFlow':
                model = tf.keras.models.load_model(path)
            else:
                model = joblib.load(path)
            acc, prec, rec, f1 = evaluate_model(model, X_test, y_test, name)
            results[name] = {'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1 Score': f1}
        else:
            print(f"Model {name} not found at {path}")

    # Print comparison table
    print("\nAlgorithm Comparison Results:")
    print("=" * 80)
    print(f"{'Algorithm':<20} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1 Score':<10}")
    print("-" * 80)
    for name, metrics in results.items():
        print(f"{name:<20} {metrics['Accuracy']:<10.4f} {metrics['Precision']:<10.4f} {metrics['Recall']:<10.4f} {metrics['F1 Score']:<10.4f}")
    print("=" * 80)

if __name__ == "__main__":
    main()
