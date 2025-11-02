import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import joblib
from pathlib import Path

def main():
    data_path = Path(__file__).resolve().parent.parent.parent / "data" / "historical_goes_2010_2015_parsed.csv"
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

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    print("XGBoost Classifier Results:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred, average='weighted'):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred, average='weighted'):.4f}")
    print(f"F1 Score: {f1_score(y_test, y_pred, average='weighted'):.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    model_save_path = Path(__file__).resolve().parent.parent.parent / "models" / "xgboost_model.joblib"
    joblib.dump(model, model_save_path)
    print(f"Model saved to {model_save_path}")

if __name__ == "__main__":
    main()
