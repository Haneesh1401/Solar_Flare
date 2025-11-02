import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import joblib
from pathlib import Path

def main():
    data_path = Path(__file__).resolve().parent.parent / "data" / "historical_goes_2010_2015_parsed.csv"
    df = pd.read_csv(data_path)

    df = df.dropna(subset=['flare_class', 'flux', 'start'])
    df['flux'] = pd.to_numeric(df['flux'], errors='coerce')
    df = df.dropna(subset=['flux'])
    df['start'] = pd.to_datetime(df['start'], errors='coerce')
    df = df.dropna(subset=['start'])
    df['peak'] = pd.to_datetime(df['peak'], errors='coerce')
    df['end'] = pd.to_datetime(df['end'], errors='coerce')
    df = df.dropna(subset=['peak', 'end'])

    # Feature engineering
    df['month'] = df['start'].dt.month
    df['day'] = df['start'].dt.day
    df['year'] = df['start'].dt.year
    df['duration_minutes'] = (df['end'] - df['start']).dt.total_seconds() / 60.0
    df['rise_minutes'] = (df['peak'] - df['start']).dt.total_seconds() / 60.0
    df['decay_minutes'] = (df['end'] - df['peak']).dt.total_seconds() / 60.0

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

    # Features
    X = df[['flux', 'month', 'day', 'year', 'duration_minutes', 'rise_minutes', 'decay_minutes']]
    y = df['flare_class_num']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Hyperparameter tuning
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    model = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(X_train_scaled, y_train)

    best_model = grid_search.best_estimator_
    print(f"Best parameters: {grid_search.best_params_}")

    y_pred = best_model.predict(X_test_scaled)

    print("Improved RandomForest Classifier Results:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred, average='weighted'):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred, average='weighted'):.4f}")
    print(f"F1 Score: {f1_score(y_test, y_pred, average='weighted'):.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    model_save_path = Path(__file__).resolve().parent.parent / "models" / "improved_random_forest.joblib"
    joblib.dump(best_model, model_save_path)
    print(f"Model saved to {model_save_path}")

if __name__ == "__main__":
    main()
